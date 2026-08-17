"""Host-backed MoE expert offload: static S-slot device cache for vLLM TPU.

Each MoE layer's full expert bank (all N experts, GMM_TP processed layout)
is mirrored in host RAM. A fixed set of S device slots caches the experts
the current batch actually routes to; ``route()`` runs the top-k on host,
ensures the chosen experts are resident (evicting LRU victims the current
batch does NOT need), and builds a [T, S] gating matrix with real logits
at slot positions and -inf elsewhere. The GMM kernel renormalizes over the
selected top-k, so the S-column remap is numerically identical to full-bank
routing when the routed experts are resident (falsified in
scripts/expert_offload_stage4.py).

Slot 0 is reserved for num_valid_tokens padding routing (expert 0, never
evicted). If a batch needs more unique experts than S, routing raises
RuntimeError instead of silently corrupting gating (capacity-first).

Wiring (stage4_patch_spec.md Section 3):
  - unquantized.py process_weights_after_loading: register_bank() per layer
    after process_unquantized_moe_weights, skipping full shard_moe_weights.
  - interface/moe.py vllm_moe_apply: get_bank(layer.layer_name) interception
    -> host topk -> route() -> replicated [T, S] gating -> slot weights.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Registry of per-layer host banks, keyed by the vLLM FusedMoE prefix
# "model.layers.{i}.experts" (== layer.layer_name at serve time).
_BANKS: dict[str, "_LayerBank"] = {}
# Slot 0 is reserved for num_valid_tokens padding routing (expert 0, never
# evicted).
_PADDING_SLOT = 0


def offload_enabled() -> bool:
    """Master switch: MOE_EXPERT_OFFLOAD (bool, default False)."""
    return bool(envs.MOE_EXPERT_OFFLOAD)


def slot_count() -> int:
    """Device slot cache size per layer: MOE_EXPERT_OFFLOAD_SLOTS (int, default 16)."""
    return int(envs.MOE_EXPERT_OFFLOAD_SLOTS)


def layer_enabled(layer_name: str) -> bool:
    """Layer allowlist: MOE_EXPERT_OFFLOAD_LAYERS (str, "" = all layers).

    Non-empty value is a comma-separated list of layer-name prefixes
    (e.g. "model.layers.0.experts,model.layers.1.experts"). Registry keys are
    the vLLM FusedMoE prefix "model.layers.{i}.experts".

    Also gated on the master switch, so layer_enabled() implies
    offload_enabled() and register_bank() (which returns None iff the master
    switch is off) can never hand the caller a None bank from this path
    (stage4_patch_spec.md gap G11).
    """
    if not offload_enabled():
        return False
    layers = envs.MOE_EXPERT_OFFLOAD_LAYERS
    if not layers:
        return True
    return any(layer_name.startswith(p) for p in layers.split(",") if p)


def get_bank(layer_name: str) -> "_LayerBank | None":
    return _BANKS.get(layer_name)


def clear_all() -> None:
    _BANKS.clear()


class _LayerBank:
    """Static S-slot device cache for one layer's expert weights.

    The full bank (all N experts of the layer, in GMM_TP processed layout)
    is mirrored in host RAM (w13_host / w2_host); only S fixed slots are
    resident on device. Slot contents can be refreshed with different expert
    values without recompiling the jitted fused_moe_func (same shapes).
    """

    def __init__(self, layer_name: str, w13_host: np.ndarray,
                 w2_host: np.ndarray, dev_w13_sharding, dev_w2_sharding):
        self.layer_name = layer_name
        self.w13_host = w13_host            # host numpy mirror [N, 2048, 2048]
        self.w2_host = w2_host              # host numpy mirror [N, 2048, 512]
        self.dev_w13_sharding = dev_w13_sharding   # GMM_TP: P(None, None, MLP_TENSOR)
        self.dev_w2_sharding = dev_w2_sharding     # GMM_TP: P(None, MLP_TENSOR, None)
        self.slots = slot_count()           # MOE_EXPERT_OFFLOAD_SLOTS (16 default)
        self._allocate_initial_slots()
        logger.info(
            "[expert-offload] layer %s: host bank %d experts, %d device slots "
            "(%.2f GB host)",
            self.layer_name, self.w13_host.shape[0], self.slots,
            (self.w13_host.nbytes + self.w2_host.nbytes) / 1e9)

    def _allocate_initial_slots(self) -> None:
        """Initial residency: experts 0..S-1 fill the S device slots.

        Slot 0 holds expert 0 (the reserved padding expert); the remaining
        slots hold experts 1..S-1. S*PER_EXP math: S=16 -> 2*16*10.486MB =
        0.3355 GB ("slots 0.34 GB" in the stage-4 falsification output).
        """
        S = self.slots
        self.slot13_host = np.array(self.w13_host[:S]).copy()
        self.slot2_host = np.array(self.w2_host[:S]).copy()
        self.slot_w13 = jax.device_put(self.slot13_host,
                                       self.dev_w13_sharding)
        self.slot_w2 = jax.device_put(self.slot2_host, self.dev_w2_sharding)
        self.slot_w13.block_until_ready()
        self.slot_w2.block_until_ready()
        self.slot_to_expert = list(range(S))       # slot index -> expert id
        self.expert_to_slot = {e: s for s, e in enumerate(self.slot_to_expert)}
        self.lru = list(range(S))                  # expert ids, LRU order

    def ensure_resident(self, expert_ids: np.ndarray) -> None:
        """Guarantee all expert_ids are resident; evict + load misses.

        Never evicts an expert that is still needed by the current batch, and
        never evicts slot _PADDING_SLOT (expert 0 stays resident). If the
        needed set exceeds the slot capacity, raises (caller should treat as
        capacity failure rather than silently corrupt routing).
        """
        needed = set(int(e) for e in np.unique(expert_ids))
        if not needed:
            return
        resident = set(self.expert_to_slot)
        misses = needed - resident
        if not misses:
            for e in needed:
                self._touch(e)
            return
        # Victims: resident experts NOT needed by this batch (LRU order),
        # excluding the reserved padding slot.
        victims = [
            e for e in self.lru if e in resident and e not in needed
            and self.expert_to_slot[e] != _PADDING_SLOT
        ]
        # Include never-touched residents not in LRU as a fallback.
        for e in sorted(resident - needed):
            if (len(victims) >= len(misses)
                    or self.expert_to_slot[e] == _PADDING_SLOT):
                continue
            if e not in victims:
                victims.append(e)
        if len(victims) < len(misses):
            raise RuntimeError(
                f"[expert-offload] layer {self.layer_name}: need "
                f"{len(misses)} slots for {len(needed)} unique experts but only "
                f"{len(victims)} evictable (S={self.slots}). Increase "
                f"MOE_EXPERT_OFFLOAD_SLOTS or reduce routing fan-out.")
        for e in misses:
            victim_expert = victims.pop(0)
            self._load_one(self.expert_to_slot[victim_expert], e)
            self._touch(e)
        for e in needed:
            self._touch(e)

    def _load_one(self, slot: int, expert_id: int) -> None:
        """Replace slot contents with a different expert, then push to device."""
        self.slot13_host[slot] = self.w13_host[expert_id]
        self.slot2_host[slot] = self.w2_host[expert_id]
        self.slot_w13 = jax.device_put(self.slot13_host, self.dev_w13_sharding)
        self.slot_w2 = jax.device_put(self.slot2_host, self.dev_w2_sharding)
        self.slot_w13.block_until_ready()
        self.slot_w2.block_until_ready()
        old = int(self.slot_to_expert[slot])
        self.expert_to_slot.pop(old, None)
        if old in self.lru:
            self.lru.remove(old)
        self.slot_to_expert[slot] = expert_id
        self.expert_to_slot[expert_id] = slot

    def _touch(self, expert_id: int) -> None:
        if expert_id in self.lru:
            self.lru.remove(expert_id)
        self.lru.append(expert_id)

    def slot_weights(self):
        return self.slot_w13, self.slot_w2

    def route(self, router_logits: np.ndarray,
              top_k: int) -> np.ndarray:
        """Host topk -> ensure resident -> build remapped [T, S] gating."""
        T, E = router_logits.shape
        unique_ids: set[int] = set()
        for t in range(T):
            top_ids = np.argpartition(router_logits[t], -top_k)[-top_k:]
            unique_ids.update(int(e) for e in top_ids)
        if unique_ids:
            self.ensure_resident(np.fromiter(unique_ids, dtype=np.int64))
        gating = np.full((T, self.slots), -np.inf, dtype=np.float32)
        for t in range(T):
            top_ids = np.argpartition(router_logits[t], -top_k)[-top_k:]
            for e in top_ids:
                s = self.expert_to_slot.get(int(e))
                if s is not None:
                    gating[t, s] = router_logits[t, int(e)]
        return gating


def register_bank(layer_name: str, w13_host: jax.Array, w2_host: jax.Array,
                  dev_w13_sharding, dev_w2_sharding) -> _LayerBank | None:
    """Register a layer's host bank; returns the bank or None if disabled."""
    if not offload_enabled():
        return None
    bank = _LayerBank(layer_name,
                      np.asarray(jax.device_get(w13_host)),
                      np.asarray(jax.device_get(w2_host)),
                      dev_w13_sharding, dev_w2_sharding)
    _BANKS[layer_name] = bank
    return bank