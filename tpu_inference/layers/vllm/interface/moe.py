# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from vllm.forward_context import is_forward_context_available
from vllm.model_executor.layers import fused_moe as vllm_fused_moe
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import \
    get_layer_from_name

# TODO: Remove this fallback after the vLLM LKG exports FusedMoEFactory.
if hasattr(vllm_fused_moe, "FusedMoEFactory"):
    FusedMoEFactory = vllm_fused_moe.FusedMoEFactory
else:
    FusedMoEFactory = vllm_fused_moe.FusedMoE

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
from tpu_inference.layers.common.sharding import is_attn_dp
from tpu_inference.layers.vllm import expert_offload
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _bank_gating_device(topk_vals: jax.Array, topk_ids: jax.Array,
                        slot_table_jnp: jax.Array, slots: int) -> jax.Array:
    """Build [T,S] gating on device via scatter from top-k device values.

    Maps each top-k expert ID to its slot via slot_table, then scatters
    the corresponding logit value into the gating output.
    """
    T, K = topk_ids.shape
    slots_for_experts = slot_table_jnp[topk_ids]  # [T,K] — slot per (t,k)
    gating = jnp.full((T, slots), -jnp.inf, dtype=jnp.float32)
    t_idx = jnp.arange(T, dtype=jnp.int32)[:, None]
    gating = gating.at[t_idx, slots_for_experts].set(topk_vals)
    return gating


def select_moe_backend_from_fused_moe_config(
        moe: FusedMoEConfig) -> MoEBackend:
    """
    Select the MoE backend based on the FusedMoEConfig.

    NOTE (jacobplatin): we don't currently support DENSE_MAT or MEGABLX_GMM
    backends on the vLLM path for now.

    Args:
        moe: The FusedMoEConfig.

    Returns:
        The selected MoE backend.
    """

    if envs.USE_MOE_EP_KERNEL:
        if moe.use_ep:
            logger.info_once("[MoE]: Using fused MoE EP kernel")
            return MoEBackend.FUSED_MOE
        logger.warning_once(
            "USE_MOE_EP_KERNEL=1 but expert parallelism is not "
            "enabled. Falling back to gmm implementation.")

    if moe.use_ep:
        logger.info_once("[MoE]: Using GMM EP kernel")
        return MoEBackend.GMM_EP

    # Use default implementation.
    logger.info_once("[MoE]: Using GMM TP kernel")
    return MoEBackend.GMM_TP


def vllm_moe_apply(layer: RoutedExperts,
                   weights: FusedMoEWeights,
                   quant_method_instance: FusedMoEMethodBase,
                   x: torch.Tensor,
                   router_logits: torch.Tensor,
                   input_ids: torch.Tensor | None = None) -> torch.Tensor:
    """
    Shared function for applying a FusedMoE layer for the TorchAX/vLLM backend.

    Args:
        layer: The FusedMoE layer.
        weights: The FusedMoE weights.
        quant_method_instance: The quantization method instance.
        x: The input tensor.
        router_logits: The router logits.

    Returns:
        The output tensor from the MoE fowrard pass.
    """
    assert isinstance(layer, RoutedExperts)
    assert isinstance(quant_method_instance, FusedMoEMethodBase)
    assert isinstance(weights, FusedMoEWeights)

    from tpu_inference.models.vllm.vllm_model_wrapper_context import \
        get_vllm_model_wrapper_context
    try:
        context = get_vllm_model_wrapper_context()
        vllm_config = context.vllm_config
    except AssertionError:
        vllm_config = None

    enable_return_routed_experts = vllm_config.model_config.enable_return_routed_experts if vllm_config else False

    if enable_return_routed_experts:
        if isinstance(router_logits, torch.Tensor):
            _, expert_indices = torch.topk(router_logits, layer.top_k, dim=-1)
            from tpu_inference.models.vllm.vllm_model_wrapper_context import \
                get_vllm_model_wrapper_context
            try:
                context = get_vllm_model_wrapper_context()
                context.expert_indices_list.append(jax_view(expert_indices))
            except AssertionError:
                pass

    mesh = quant_method_instance.mesh
    is_dp = is_attn_dp(mesh)

    extra_kwargs = dict(quant_method_instance.extra_backend_kwargs)
    extra_kwargs["scatter_results"] = is_dp

    # Defer the tensor-parallel all-reduce inside the GMM kernel exactly when the
    # runner does NOT expect the fused output to be reduced -- the deferred path
    # where the shared and fused outputs are summed and reduced together in a
    # single collective downstream. This is the inverse of (and tied to)
    # ``VllmMoERunner._fused_output_is_reduced`` so the two never drift.
    if is_forward_context_available():
        runner = get_layer_from_name(layer.layer_name)
        extra_kwargs["defer_all_reduce"] = not runner._fused_output_is_reduced

    if getattr(layer, "hash_indices_table", None) is not None:
        assert input_ids is not None, "input_ids must be provided when hash_indices_table is present in the layer"
        hash_table = layer.hash_indices_table
        hash_based_topk_indices = jax_view(hash_table)[jax_view(input_ids)]
        extra_kwargs["hash_based_topk_indices"] = hash_based_topk_indices

    if getattr(layer, "e_score_correction_bias", None) is not None:
        extra_kwargs["e_score_correction_bias"] = jax_view(
            layer.e_score_correction_bias)

    # Route padding tokens to a single expert instead of activating unnecessary
    # experts. Applicable when DP attention size is 1 (pure TP attention, e.g.
    # TP8_EP), since with DP attention the padding for each rank is interleaved.
    if envs.MOE_ROUTE_PADDING_TO_EXPERT0 and not is_dp:
        try:
            from vllm.forward_context import get_forward_context
            attn_meta = get_forward_context().attn_metadata
            if isinstance(attn_meta, dict):
                attn_meta = next(iter(attn_meta.values()))
            qsl = getattr(attn_meta, "query_start_loc", None)
            if qsl is not None:
                if isinstance(qsl, torch.Tensor):
                    qsl = jax_view(qsl)
                extra_kwargs["num_valid_tokens"] = qsl[-1]
        except Exception as e:
            logger.warning_once(
                "MOE_ROUTE_PADDING_TO_EXPERT0: failed to read num_valid_tokens "
                "from attn metadata, skipping padding routing (%s)", e)

    # Host-backed MoE expert offload: if this layer has a registered host
    # bank, route on host (topk -> ensure resident -> [T,S] remap), place the
    # gating replicated, and compute with the S-slot device bank. Quantized
    # (MXFP4/FP4) banks feed their slot block scales alongside the packed
    # weights (the kernel needs fp32 scales to dequantize); unquantized banks
    # carry slot_w13_scale / slot_w2_scale == None and the feed falls back to
    # scale=None exactly as before the scale extension. Bias stays None (the
    # offload gates refuse layers with bias).
    bank = expert_offload.get_bank(layer.layer_name)
    if bank is not None:
        # Device-first hit path: top_k on device, no device_get on [T,E] logits.
        # Slot table built from bank attrs (resides on device after ensure_resident scatter).
        slot_table_jnp = jnp.array(bank.slot_to_expert, dtype=jnp.int32)
        topk_vals, topk_ids = jax.lax.top_k(jax_view(router_logits), layer.top_k)
        # Export tiny [T,K] int32 IDs for host-side bookkeeping (miss/over-footprint)
        topk_ids_np = np.asarray(jax.device_get(jax_view(topk_ids))).astype(np.int64)
        unique_ids_np = np.unique(topk_ids_np)
        # Over-footprint: unique experts exceed slot capacity → fallback to legacy wave path
        if len(unique_ids_np) > bank.slots - 1:
            logger.info_once(
                "[MoE]: %s over-footprint %d unique > S-1=%d, falling back to legacy route()",
                layer.layer_name, len(unique_ids_np), bank.slots - 1)
            logits_np = np.asarray(jax.device_get(jax_view(router_logits)))
            g_np = bank.route(logits_np, layer.top_k)
            g = jax.device_put(g_np, NamedSharding(mesh, PartitionSpec()))
            weights = FusedMoEWeights(
                w13_weight=bank.slot_w13,
                w13_weight_scale=bank.slot_w13_scale,
                w13_bias=None,
                w2_weight=bank.slot_w2,
                w2_weight_scale=bank.slot_w2_scale,
                w2_bias=None,
            )
            return torch_view(
                moe_apply(
                    layer=layer, x=jax_view(x), gating_output=g,
                    weights=weights, moe_backend=quant_method_instance.moe_backend,
                    mesh=quant_method_instance.mesh,
                    extra_backend_kwargs=extra_kwargs,
                ))
        # Residency check on host
        is_resident = np.all(np.isin(unique_ids_np, bank.slot_to_expert))
        if not is_resident:
            bank.ensure_resident(topk_ids_np, slot_table=slot_table_jnp)
            slot_table_jnp = jnp.array(bank.slot_to_expert, dtype=jnp.int32)
            # topk_ids unchanged (same router_logits), but slot_table updated —
            # re-run top_k for fresh slot mapping
            topk_vals, topk_ids = jax.lax.top_k(jax_view(router_logits), layer.top_k)
        # All resident → device-built gating [T,S] via scatter
        g = _bank_gating_device(topk_vals, topk_ids, slot_table_jnp, bank.slots)
        weights = FusedMoEWeights(
            w13_weight=bank.slot_w13,
            w13_weight_scale=bank.slot_w13_scale,
            w13_bias=None,
            w2_weight=bank.slot_w2,
            w2_weight_scale=bank.slot_w2_scale,
            w2_bias=None,
        )
        return torch_view(
            moe_apply(
                layer=layer, x=jax_view(x), gating_output=g,
                weights=weights, moe_backend=quant_method_instance.moe_backend,
                mesh=quant_method_instance.mesh,
                extra_backend_kwargs=extra_kwargs,
            ))

    return torch_view(
        moe_apply(
            layer=layer,
            x=jax_view(x),
            gating_output=jax_view(router_logits),
            weights=weights,
            moe_backend=quant_method_instance.moe_backend,
            mesh=quant_method_instance.mesh,
            extra_backend_kwargs=extra_kwargs,
        ))
