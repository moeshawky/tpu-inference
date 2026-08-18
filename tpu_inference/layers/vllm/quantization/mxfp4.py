# Copyright 2025 Google LLC
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

from typing import Optional

import gc

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig, mxfp4_w4a16_moe_quant_config)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.mxfp4 import \
    GptOssMxfp4Config as Mxfp4Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference.layers.common.moe import \
    FusedMoEMethodBase as TpuFusedMoEMethodBase
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, quantize_moe_weights,
    shard_moe_weights)
from tpu_inference.layers.common.quant_methods import MXFP4
from tpu_inference.layers.common.quantization import (
    MXFP4_REQUANTIZED_BLOCK_SIZE, dequantize_tensor_from_mxfp4_packed)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.layers.vllm import expert_offload
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, t2j

P = PartitionSpec

logger = init_logger(__name__)


@register_quantization_config(MXFP4)
class VllmMxfp4Config(Mxfp4Config, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return MXFP4

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:

        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if self.ignored_layers and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return VllmUnquantizedLinearMethod(linear_config)
            logger.warning_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.")
            return VllmUnquantizedLinearMethod(linear_config)
        elif isinstance(layer, RoutedExperts):
            moe_config = self.get_moe_config(layer)
            return VllmMxfp4MoEMethod(moe_config, self.mesh)
        elif isinstance(layer, Attention):
            logger.warning_once("MXFP4 attention layer is not implemented. "
                                "Skipping quantization for this layer.")
        return None


class VllmMxfp4MoEMethod(Mxfp4MoEMethod, FusedMoEMethodBase):

    def __init__(
        self,
        moe: FusedMoEConfig,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        FusedMoEMethodBase.__init__(self, moe)

        # We piggyback on triton implementation as it applies minimal hardware
        # specific post processing to the weights.
        self.mxfp4_backend = Mxfp4MoeBackend.TRITON

        # vLLM's Mxfp4MoEMethod.__init__ (intentionally skipped above) sets
        # is_k3_situ_aiter for the ROCm AITER Kimi-K3 path, and inherited
        # methods such as create_weights read it. That path never applies on
        # TPU, so pin it to False.
        self.is_k3_situ_aiter = False

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        TpuFusedMoEMethodBase.__init__(self, self.moe_backend, ep_axis_name)

    def _gmm_tp_w13_sharding(self) -> NamedSharding:
        """GMM_TP w13 layout: shard the last (MLP_TENSOR) axis."""
        return NamedSharding(self.mesh,
                             P(None, None, ShardingAxisName.MLP_TENSOR))

    def _gmm_tp_w2_sharding(self) -> NamedSharding:
        """GMM_TP w2 layout: shard the middle (MLP_TENSOR) axis."""
        return NamedSharding(self.mesh,
                             P(None, ShardingAxisName.MLP_TENSOR, None))

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    @property
    def is_monolithic(self) -> bool:
        return True

    def _gmm_tp_w13_scale_sharding(self) -> NamedSharding:
        """GMM_TP w13 block-scale layout: shard the last (MLP_TENSOR) axis.

        4-D spec the kernel consumes (fused_moe_gmm.py tensor_parallel_gmm):
        P(None, None, None, MLP_TENSOR) on [E, 16, 1, 4096] -- NOT the 2-D
        weight sharding (P(None, None, MLP_TENSOR)); the scale contract is
        dimensionally different from the weight contract.
        """
        return NamedSharding(
            self.mesh,
            P(None, None, None, ShardingAxisName.MLP_TENSOR))

    def _gmm_tp_w2_scale_sharding(self) -> NamedSharding:
        """GMM_TP w2 block-scale layout: shard the first block axis.

        4-D spec the kernel consumes (fused_moe_gmm.py tensor_parallel_gmm):
        P(None, MLP_TENSOR, None, None) on [E, 8, 1, 4096] -- NOT the 2-D
        weight sharding (P(None, MLP_TENSOR, None)). NOTE: 8 blocks over the
        MLP_TENSOR axis must divide evenly (mesh axis size 4 or 2); verified
        at load time on TPU.
        """
        return NamedSharding(
            self.mesh,
            P(None, ShardingAxisName.MLP_TENSOR, None, None))

    def process_weights_after_loading(self, layer: torch.nn.Module):
        assert isinstance(layer, RoutedExperts)
        has_bias = layer.moe_config.has_bias
        offload_layer = (not has_bias
                         and expert_offload.layer_enabled(layer.layer_name))
        if offload_layer:
            source_bytes = sum(
                expert_offload.tensor_nbytes(tensor) for tensor in (
                    layer.w13_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight,
                    layer.w2_weight_scale,
                ))
            expert_offload.check_host_memory_budget(layer.layer_name,
                                                    source_bytes)

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w13_bias = t2j(layer.w13_bias, use_dlpack=False) if has_bias else None

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)
        w2_bias = t2j(layer.w2_bias, use_dlpack=False) if has_bias else None

        @jax.jit
        def process_mxfp4_moe_weights(
            w13_weight: jax.Array,
            w13_weight_scale: jax.Array,
            w13_bias: jax.Array | None,
            w2_weight: jax.Array,
            w2_weight_scale: jax.Array,
            w2_bias: jax.Array | None,
        ) -> FusedMoEWeights:
            # Dequantize fp4 weights into fp32.
            w13_weight = dequantize_tensor_from_mxfp4_packed(
                w13_weight, w13_weight_scale, 2, jnp.float32)
            w2_weight = dequantize_tensor_from_mxfp4_packed(
                w2_weight, w2_weight_scale, 2, jnp.float32)
            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            weights = quantize_moe_weights(
                FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=None,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=None,
                    w2_bias=w2_bias,
                ),
                jnp.float4_e2m1fn,
                MXFP4_REQUANTIZED_BLOCK_SIZE,
                w13_interleave=w13_interleave,
            )
            return process_moe_weights(
                weights,
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        # Keep the large dequantized expert tensors off device: the
        # dequant/requant pipeline needs ~42 GB of HLO temporaries on a
        # single chip when run on TPU (RESOURCE_EXHAUSTED on v5e 16 GB).
        # Mirror the JAX-side MXFP4 path (layers/jax/quantization/mxfp4.py),
        # which runs this pipeline under the CPU mesh. t2j commits to the
        # TPU mesh, so re-place the inputs on the CPU device inside the
        # context (jit rejects arguments on a different platform).
        with cpu_mesh_context():
            cpu_device = jax.devices("cpu")[0]
            w13_weight = jax.device_put(w13_weight, cpu_device)
            w13_weight_scale = jax.device_put(w13_weight_scale, cpu_device)
            if w13_bias is not None:
                w13_bias = jax.device_put(w13_bias, cpu_device)
            w2_weight = jax.device_put(w2_weight, cpu_device)
            w2_weight_scale = jax.device_put(w2_weight_scale, cpu_device)
            if w2_bias is not None:
                w2_bias = jax.device_put(w2_bias, cpu_device)

            weights = process_mxfp4_moe_weights(
                w13_weight,
                w13_weight_scale,
                w13_bias,
                w2_weight,
                w2_weight_scale,
                w2_bias,
            )

        if offload_layer:
            # Host-backed expert offload (MXFP4/FP4 path): register the full
            # host bank -- packed float4_e2m1fn weights AND their fp32 block
            # scales -- and keep only the initial S-slot bank on device, then
            # skip full shard_moe_weights (mirror of the unquantized branch
            # in unquantized.py, extended with the scale contract: the kernel
            # cannot dequantize packed fp4 without the fp32 block scales).
            # Hash-routed layers are refused inside register_bank (shared
            # choke point covering this gate and the unquantized gate), so a
            # None bank here falls through to the full shard path below.
            bank = expert_offload.register_bank(
                layer.layer_name, weights.w13_weight, weights.w2_weight,
                self._gmm_tp_w13_sharding(), self._gmm_tp_w2_sharding(),
                w13_scale_host=weights.w13_weight_scale,
                w2_scale_host=weights.w2_weight_scale,
                dev_w13_scale_sharding=self._gmm_tp_w13_scale_sharding(),
                dev_w2_scale_sharding=self._gmm_tp_w2_scale_sharding(),
                layer=layer)
            if bank is not None:
                layer.w13_weight = Parameter(torch_view(bank.slot_w13),
                                             requires_grad=False)
                layer.w2_weight = Parameter(torch_view(bank.slot_w2),
                                            requires_grad=False)
                layer.w13_weight_scale = Parameter(
                    torch_view(bank.slot_w13_scale), requires_grad=False)
                layer.w2_weight_scale = Parameter(
                    torch_view(bank.slot_w2_scale), requires_grad=False)
                jax.effects_barrier()
                del w13_weight, w13_weight_scale, w13_bias
                del w2_weight, w2_weight_scale, w2_bias, weights
                gc.collect()
                return

        # The pipeline above ran on the CPU mesh; commit the processed
        # weights back to the TPU mesh (replicated) so shard_moe_weights'
        # reshard accepts them -- jax.device_put rejects inputs committed
        # to a different platform or a partial device subset.
        weights = jax.tree_util.tree_map(
            lambda a: jax.device_put(a, NamedSharding(self.mesh, P())),
            weights)

        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)

        if has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

        # Do not let async CPU/XLA intermediates accumulate across the many
        # DeepSeek MoE layers. The guard above prevents the next layer from
        # starting when its working set would cross the cgroup budget; this
        # barrier and explicit release keep the accepted layers below it.
        jax.effects_barrier()
        del w13_weight, w13_weight_scale, w13_bias
        del w2_weight, w2_weight_scale, w2_bias, weights
        gc.collect()

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        has_bias = layer.moe_config.has_bias
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_bias=jax_view(layer.w13_bias) if has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_bias=jax_view(layer.w2_bias) if has_bias else None,
        )

        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits,
                              input_ids=input_ids)
