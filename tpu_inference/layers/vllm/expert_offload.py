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

Quantized (MXFP4/FP4) banks additionally mirror the fp32 block scales
(w13_weight_scale / w2_weight_scale, 4-D GMM_TP processed layout) next to
packed float4_e2m1fn weights; for the DeepSeek-V4-0731 path these are
[N, 16, 1, 4096] / [N, 8, 1, 4096]. The slot cache carries both so an
expert's scales are always swapped together with its weights. The kernel
consumes the 4-D scale shardings P(None,None,None,MLP_TENSOR) and
P(None,MLP_TENSOR,None,None) (fused_moe_gmm.py tensor_parallel_gmm), which
differ from the 2-D weight shardings.

Slot 0 is reserved for num_valid_tokens padding routing (expert 0, never
evicted). If a batch needs more unique experts than S, routing raises
RuntimeError instead of silently corrupting gating (capacity-first).

Registration refuses hash-routed layers (any layer carrying
``hash_indices_table``, e.g. DeepSeek-V4's layers 0-2): banking them would
feed S-column gating to fused_moe_gmm.py together with hash indices in
expert-id space [0, num_experts), an out-of-bounds take_along_axis. Hash
layers stay resident on the full-bank path.

Wiring (stage4_patch_spec.md Section 3 + mxfp4 offload):
  - unquantized.py process_weights_after_loading: register_bank() per layer
    after process_unquantized_moe_weights, skipping full shard_moe_weights
    (no scales).
  - mxfp4.py process_weights_after_loading: register_bank() per layer after
    the processed packed fp4 weights + fp32 block scales exist, skipping
    full shard_moe_weights (scales included).
  - interface/moe.py vllm_moe_apply: get_bank(layer.layer_name) interception
    -> host topk -> route() -> replicated [T, S] gating -> slot weights and
    slot scales.
"""

import hashlib
import mmap as _mmap
import os
import re
import struct
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_BYTES_PER_GIB = 1 << 30
_CGROUP_MEMORY_CURRENT = (
    "/sys/fs/cgroup/memory.current",
    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
)
_CGROUP_MEMORY_LIMIT = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)


def _read_memory_counter(paths: tuple[str, ...]) -> int | None:
    """Read the first available Linux memory counter, tolerating ``max``."""
    for path in paths:
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if not value or value == "max":
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _read_proc_mem_available() -> int | None:
    """Read host MemAvailable without importing a third-party monitor."""
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            try:
                # /proc/meminfo reports this field in KiB.
                return int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def _read_memory_stat() -> dict[str, int]:
    """Read cgroup v2 memory.stat counters when available."""
    try:
        lines = Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
    except OSError:
        return {}
    counters: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 2:
            try:
                counters[fields[0]] = int(fields[1])
            except ValueError:
                continue
    return counters


def _host_memory_snapshot() -> dict[str, int | None]:
    """Return committed and reclaimable cgroup memory separately.

    Kaggle exposes more physical RAM through /proc than the notebook cgroup
    permits. The cgroup limit remains authoritative, but ``memory.current``
    includes reclaimable NFS page cache. Treating every cached checkpoint page
    as committed process memory rejects valid loads; report that cache
    separately and base admission on committed memory plus an explicit reserve.
    """
    current = _read_memory_counter(_CGROUP_MEMORY_CURRENT)
    limit = _read_memory_counter(_CGROUP_MEMORY_LIMIT)
    stats = _read_memory_stat()
    file_bytes = stats.get("file", 0)
    shmem_bytes = stats.get("shmem", 0)
    unevictable_bytes = stats.get("unevictable", 0)
    reclaimable_file = max(file_bytes - shmem_bytes - unevictable_bytes, 0)
    committed = (max(current - reclaimable_file, 0)
                 if current is not None else None)
    cgroup_available = (max(limit - committed, 0)
                        if committed is not None and limit is not None else None)
    proc_available = _read_proc_mem_available()
    candidates = [value for value in (cgroup_available, proc_available)
                  if value is not None]
    available = min(candidates) if candidates else None
    return {
        "cgroup_current": current,
        "cgroup_limit": limit,
        "cgroup_available": cgroup_available,
        "cgroup_committed": committed,
        "cgroup_reclaimable_file": reclaimable_file,
        "proc_available": proc_available,
        "available": available,
    }


def _memory_guard_message(available: int | None, working_set: int,
                          reserve: int) -> str | None:
    """Return an admission error when the next layer cannot fit safely."""
    if available is None or available >= working_set + reserve:
        return None
    return (f"host memory guard refused next MoE layer: available="
            f"{available / _BYTES_PER_GIB:.2f} GiB, estimated CPU working "
            f"set={working_set / _BYTES_PER_GIB:.2f} GiB, safety reserve="
            f"{reserve / _BYTES_PER_GIB:.2f} GiB")


def tensor_nbytes(tensor) -> int:
    """Return tensor bytes without materializing or copying the tensor."""
    nbytes = getattr(tensor, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return int(tensor.numel()) * int(tensor.element_size())


def check_host_memory_budget(layer_name: str, source_bytes: int) -> None:
    """Reject a layer before its CPU MXFP4 peak can kill the notebook.

    The CPU requantization path has a large transient working set and bank
    registration may briefly hold both the processed JAX arrays and NumPy
    views. Estimate at least the configured working set, scaling it with the
    incoming tensor size for larger models, and preserve an explicit reserve
    for the notebook kernel and runtime. This is an admission guard, not a
    claim that the model will fit after admission.
    """
    if not envs.MOE_EXPERT_OFFLOAD_HOST_MEMORY_GUARD:
        return
    if source_bytes < 0:
        raise ValueError(f"negative source_bytes for {layer_name}: {source_bytes}")

    configured_working_set = max(
        int(envs.MOE_EXPERT_OFFLOAD_CPU_WORKING_SET_GIB), 0) * _BYTES_PER_GIB
    working_set = max(configured_working_set, source_bytes * 8)
    reserve = max(int(envs.MOE_EXPERT_OFFLOAD_HOST_MEMORY_RESERVE_GIB),
                  0) * _BYTES_PER_GIB
    snapshot = _host_memory_snapshot()
    available = snapshot["available"]
    logger.info(
        "[expert-offload] host memory admission layer=%s available=%.2f GiB "
        "cgroup_current=%.2f GiB committed=%.2f GiB reclaimable_file=%.2f "
        "GiB cgroup_limit=%.2f GiB source=%.2f GiB working_set=%.2f GiB "
        "reserve=%.2f GiB",
        layer_name,
        available / _BYTES_PER_GIB if available is not None else -1.0,
        snapshot["cgroup_current"] / _BYTES_PER_GIB
        if snapshot["cgroup_current"] is not None else -1.0,
        snapshot["cgroup_committed"] / _BYTES_PER_GIB
        if snapshot["cgroup_committed"] is not None else -1.0,
        snapshot["cgroup_reclaimable_file"] / _BYTES_PER_GIB,
        snapshot["cgroup_limit"] / _BYTES_PER_GIB
        if snapshot["cgroup_limit"] is not None else -1.0,
        source_bytes / _BYTES_PER_GIB,
        working_set / _BYTES_PER_GIB,
        reserve / _BYTES_PER_GIB,
    )
    message = _memory_guard_message(available, working_set, reserve)
    if message is not None:
        raise RuntimeError(
            f"[expert-offload] {message}; layer={layer_name}. Reduce "
            "MOE_EXPERT_OFFLOAD_LAYERS, lower MOE_EXPERT_OFFLOAD_SLOTS, "
            "or increase the host-memory budget before retrying.")


# Registry of per-layer host banks, keyed by the vLLM FusedMoE prefix
# "model.layers.{i}.ffn.experts" (== layer.layer_name at serve time). The
# older comment said ".experts" -- wrong for DeepseekV4, whose FusedMoEFactory
# is created with prefix=f"{prefix}.experts" under the .ffn block
# (models/vllm/experimental/deepseek_v4.py:182). A bare "model.layers.3" form
# would false-match layer 30, so the full ".ffn.experts" prefix is required.
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
    (e.g. "model.layers.0.ffn.experts,model.layers.1.ffn.experts"). Registry
    keys are the vLLM FusedMoE prefix "model.layers.{i}.ffn.experts".

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

    For quantized (MXFP4/FP4) banks the fp32 block scales are mirrored and
    slotted alongside the packed weights (w13_scale_host / w2_scale_host,
    shapes [N, 8, 1, 4096] / [N, 4, 1, 4096] for DeepSeek-V4 0731 dims) and
    device_put with the 4-D kernel shardings the GMM kernel consumes
    (fused_moe_gmm.py tensor_parallel_gmm). Unquantized banks keep the scale
    hosts and device slots None.
    """

    def __init__(self, layer_name: str, w13_host: np.ndarray,
                 w2_host: np.ndarray, dev_w13_sharding, dev_w2_sharding,
                 w13_scale_host: np.ndarray | None = None,
                 w2_scale_host: np.ndarray | None = None,
                 dev_w13_scale_sharding=None, dev_w2_scale_sharding=None,
                 store: "_LayerStore | None" = None,
                 push_mode: str = "full"):
        self.layer_name = layer_name
        self.store = store
        if store is not None:
            # Design D store-backed bank: the canonical layer file (mmap'd,
            # sha-verified) is the persistent expert representation. No
            # anonymous full-bank arrays may be handed in; the only
            # persistent host state is the S-slot mirror (full-push mode)
            # and the optional bounded record hot cache (store side).
            if push_mode not in ("scatter", "full"):
                raise ValueError(
                    f"[expert-offload] layer {layer_name}: store-backed "
                    f"push_mode must be 'scatter' or 'full', got {push_mode!r}")
            if w13_host is not None or w2_host is not None:
                raise ValueError(
                    f"[expert-offload] layer {layer_name}: store-backed "
                    "banks must not receive anonymous full-bank arrays")
            if w13_scale_host is not None or w2_scale_host is not None:
                raise ValueError(
                    f"[expert-offload] layer {layer_name}: store-backed "
                    "banks read block scales from the store")
            self.w13_host = None
            self.w2_host = None
            self.w13_scale_host = None
            self.w2_scale_host = None
            self.push_mode = push_mode
            self._w13_packed = store.w13_row_dtype == np.uint8
            self._w2_packed = store.w2_row_dtype == np.uint8
            self._float4_dtype = (np.dtype("float4_e2m1fn")
                                  if self._w13_packed or self._w2_packed
                                  else None)
        else:
            self.w13_host = w13_host            # processed host mirror [N, ...]
            self.w2_host = w2_host              # processed host mirror [N, ...]
            self.push_mode = "full"
            # JAX float4 uses one byte per value in host NumPy storage,
            # although only the low nibble is meaningful. Pack two codes per
            # byte to keep all 40 processed banks resident without an
            # overlay/NFS write path.
            self._w13_packed = (w13_host.dtype == np.uint8
                                and envs.MOE_EXPERT_OFFLOAD_PACKED_HOST)
            self._w2_packed = (w2_host.dtype == np.uint8
                               and envs.MOE_EXPERT_OFFLOAD_PACKED_HOST)
            self._float4_dtype = (np.dtype("float4_e2m1fn")
                                  if self._w13_packed or self._w2_packed
                                  else None)
        self.dev_w13_sharding = dev_w13_sharding   # GMM_TP: P(None, None, MLP_TENSOR)
        self.dev_w2_sharding = dev_w2_sharding     # GMM_TP: P(None, MLP_TENSOR, None)
        # MXFP4/FP4 block scales (fp32, GMM_TP processed 4-D layout) mirrored
        # on host; None for unquantized banks.
        if store is None:
            self.w13_scale_host = w13_scale_host  # host mirror [N, 16, 1, 4096] or None
            self.w2_scale_host = w2_scale_host    # host mirror [N, 8, 1, 4096] or None
        # 4-D kernel scale shardings: P(None,None,None,MLP_TENSOR) /
        # P(None,MLP_TENSOR,None,None) -- NOT the 2-D weight shardings above.
        self.dev_w13_scale_sharding = dev_w13_scale_sharding
        self.dev_w2_scale_sharding = dev_w2_scale_sharding
        if store is None and (w13_scale_host is None) != (w2_scale_host is None):
            raise ValueError(
                f"[expert-offload] layer {layer_name}: w13/w2 scale hosts "
                "must be provided together (both None for unquantized).")
        if (store is None and w13_scale_host is not None
                and (dev_w13_scale_sharding is None
                     or dev_w2_scale_sharding is None)):
            raise ValueError(
                f"[expert-offload] layer {layer_name}: scale shardings "
                "required when scale hosts are provided.")
        if store is not None and store.has_scales and (
                dev_w13_scale_sharding is None
                or dev_w2_scale_sharding is None):
            raise ValueError(
                f"[expert-offload] layer {layer_name}: scale shardings "
                "required for a store that carries block scales")
        self.slots = slot_count()           # MOE_EXPERT_OFFLOAD_SLOTS (16 default)
        self._row_sharding_cache: dict[int, NamedSharding] = {}
        self._allocate_initial_slots()
        if store is None:
            host_bytes = (self.w13_host.nbytes + self.w2_host.nbytes
                          + (0 if self.w13_scale_host is None
                             else self.w13_scale_host.nbytes
                             + self.w2_scale_host.nbytes))
            logger.info(
                "[expert-offload] layer %s: host bank %d experts, %d device "
                "slots (%.2f GB host, scales %s)",
                self.layer_name, self.w13_host.shape[0], self.slots,
                host_bytes / 1e9,
                "yes" if self.w13_scale_host is not None else "no")
        else:
            mirror_bytes = 0
            if self.push_mode == "full":
                mirror_bytes = (self.slot13_host.nbytes + self.slot2_host.nbytes
                                + (0 if self.slot13_scale_host is None
                                   else self.slot13_scale_host.nbytes
                                   + self.slot2_scale_host.nbytes))
            logger.info(
                "[expert-offload] layer %s: STORE-backed %d experts, %d "
                "device slots (push=%s, %s, %.2f GB host mirror, scales %s)",
                self.layer_name, store.n_experts, self.slots, self.push_mode,
                store.path, mirror_bytes / 1e9,
                "yes" if store.has_scales else "no")

    def _row_sharding(self, sharding: NamedSharding) -> NamedSharding:
        """Row sharding for scatter updates: the slot spec minus axis 0.

        Slot arrays shard the expert axis 0 as replicated (P(None, ...));
        dropping axis 0 yields the NamedSharding a per-row device_put must
        carry so ``.at[slot].set(row)`` is a static-shape, per-device-local
        update (no MoE graph recompilation — the kernel receives the slot
        arrays as arguments with unchanged shapes/shardings).
        """
        key = id(sharding)
        cached = self._row_sharding_cache.get(key)
        if cached is None:
            cached = NamedSharding(sharding.mesh,
                                   PartitionSpec(*sharding.spec[1:]))
            self._row_sharding_cache[key] = cached
        return cached

    def _unpack_weight_rows(self, packed: np.ndarray,
                            packed_enabled: bool) -> np.ndarray:
        """Expand packed FP4 codes into the dtype consumed by the TPU kernel."""
        if not packed_enabled:
            return np.array(packed).copy()
        full_shape = list(packed.shape)
        full_shape[-1] *= 2
        raw = np.empty(full_shape, dtype=np.uint8)
        raw[..., 0::2] = packed & 0x0F
        raw[..., 1::2] = (packed >> 4) & 0x0F
        return raw.view(self._float4_dtype)

    def _allocate_initial_slots(self) -> None:
        """Initial residency: experts 0..S-1 fill the S device slots.

        Slot 0 holds expert 0 (the reserved padding expert); the remaining
        slots hold experts 1..S-1. Store-backed banks read the initial
        records from the canonical store (page-cache-hot right after the
        load-time write) instead of the anonymous full bank; scatter mode
        keeps NO host mirror (the device arrays ARE the cache), full-push
        mode keeps only the packed S-slot mirror.
        """
        S = self.slots
        if self.store is not None:
            w13_rows = np.empty((S,) + self.store.w13_shape,
                                dtype=self.store.w13_row_dtype)
            w2_rows = np.empty((S,) + self.store.w2_shape,
                               dtype=self.store.w2_row_dtype)
            s13_rows = (np.empty((S,) + self.store.s13_shape,
                                 dtype=np.float32)
                        if self.store.has_scales else None)
            s2_rows = (np.empty((S,) + self.store.s2_shape,
                                dtype=np.float32)
                       if self.store.has_scales else None)
            for e in range(S):
                w13_e, w2_e, s13_e, s2_e = self.store.read_record(e)
                w13_rows[e] = w13_e
                w2_rows[e] = w2_e
                if s13_e is not None:
                    s13_rows[e] = s13_e
                    s2_rows[e] = s2_e
            if self.push_mode == "scatter":
                # Transient S-row materialization only; the host mirror is
                # deliberately NOT retained (Design D: anon = O(hot cache)).
                self.slot_w13 = jax.device_put(
                    self._unpack_weight_rows(w13_rows, self._w13_packed),
                    self.dev_w13_sharding)
                self.slot_w2 = jax.device_put(
                    self._unpack_weight_rows(w2_rows, self._w2_packed),
                    self.dev_w2_sharding)
                self.slot_w13.block_until_ready()
                self.slot_w2.block_until_ready()
                self.slot13_host = None
                self.slot2_host = None
                if self.store.has_scales:
                    self.slot_w13_scale = jax.device_put(
                        s13_rows, self.dev_w13_scale_sharding)
                    self.slot_w2_scale = jax.device_put(
                        s2_rows, self.dev_w2_scale_sharding)
                    self.slot_w13_scale.block_until_ready()
                    self.slot_w2_scale.block_until_ready()
                    self.slot13_scale_host = None
                    self.slot2_scale_host = None
                else:
                    self.slot_w13_scale = None
                    self.slot_w2_scale = None
                    self.slot13_scale_host = None
                    self.slot2_scale_host = None
            else:
                # Full-push mode: keep the packed (storage-form) S-slot
                # mirror — 0.77 GB/layer for DeepSeek dims, half the
                # anonymous-mode unpacked mirror.
                self.slot13_host = w13_rows
                self.slot2_host = w2_rows
                self.slot_w13 = jax.device_put(
                    self._unpack_weight_rows(w13_rows, self._w13_packed),
                    self.dev_w13_sharding)
                self.slot_w2 = jax.device_put(
                    self._unpack_weight_rows(w2_rows, self._w2_packed),
                    self.dev_w2_sharding)
                self.slot_w13.block_until_ready()
                self.slot_w2.block_until_ready()
                if self.store.has_scales:
                    self.slot13_scale_host = s13_rows
                    self.slot2_scale_host = s2_rows
                    self.slot_w13_scale = jax.device_put(
                        s13_rows, self.dev_w13_scale_sharding)
                    self.slot_w2_scale = jax.device_put(
                        s2_rows, self.dev_w2_scale_sharding)
                    self.slot_w13_scale.block_until_ready()
                    self.slot_w2_scale.block_until_ready()
                else:
                    self.slot13_scale_host = None
                    self.slot2_scale_host = None
                    self.slot_w13_scale = None
                    self.slot_w2_scale = None
            self.slot_to_expert = list(range(S))
            self.expert_to_slot = {e: s for s, e in enumerate(
                self.slot_to_expert)}
            self.lru = list(range(S))
            return
        self.slot13_host = self._unpack_weight_rows(
            self.w13_host[:S], self._w13_packed)
        self.slot2_host = self._unpack_weight_rows(
            self.w2_host[:S], self._w2_packed)
        self.slot_w13 = jax.device_put(self.slot13_host,
                                       self.dev_w13_sharding)
        self.slot_w2 = jax.device_put(self.slot2_host, self.dev_w2_sharding)
        self.slot_w13.block_until_ready()
        self.slot_w2.block_until_ready()
        if self.w13_scale_host is not None:
            self.slot13_scale_host = np.array(self.w13_scale_host[:S]).copy()
            self.slot2_scale_host = np.array(self.w2_scale_host[:S]).copy()
            self.slot_w13_scale = jax.device_put(
                self.slot13_scale_host, self.dev_w13_scale_sharding)
            self.slot_w2_scale = jax.device_put(
                self.slot2_scale_host, self.dev_w2_scale_sharding)
            self.slot_w13_scale.block_until_ready()
            self.slot_w2_scale.block_until_ready()
        else:
            self.slot13_scale_host = None
            self.slot2_scale_host = None
            self.slot_w13_scale = None
            self.slot_w2_scale = None
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
        """Replace slot contents with a different expert, then push to device.

        For quantized banks the fp32 block scales are refreshed in the SAME
        host slot row and pushed with the same sharding, so a slot never
        holds expert A's weights with expert B's scales (eviction swaps the
        pair atomically). Store-backed banks read ONE record (weights +
        scales, sha-verified) from the canonical store; scatter mode pushes
        only the touched row (24.4 MiB for DeepSeek dims vs 1.5 GiB for a
        full-slot push) via per-row device_put + ``.at[slot].set``.
        """
        if self.store is not None:
            w13_row, w2_row, s13_row, s2_row = self.store.read_record(
                expert_id)
            if self.push_mode == "scatter":
                self.slot_w13 = self.slot_w13.at[slot].set(
                    jax.device_put(
                        self._unpack_weight_rows(w13_row, self._w13_packed),
                        self._row_sharding(self.dev_w13_sharding)))
                self.slot_w2 = self.slot_w2.at[slot].set(
                    jax.device_put(
                        self._unpack_weight_rows(w2_row, self._w2_packed),
                        self._row_sharding(self.dev_w2_sharding)))
                if self.store.has_scales:
                    self.slot_w13_scale = self.slot_w13_scale.at[slot].set(
                        jax.device_put(
                            np.array(s13_row),
                            self._row_sharding(self.dev_w13_scale_sharding)))
                    self.slot_w2_scale = self.slot_w2_scale.at[slot].set(
                        jax.device_put(
                            np.array(s2_row),
                            self._row_sharding(self.dev_w2_scale_sharding)))
                self.slot_w13.block_until_ready()
                self.slot_w2.block_until_ready()
                if self.store.has_scales:
                    self.slot_w13_scale.block_until_ready()
                    self.slot_w2_scale.block_until_ready()
            else:
                # Full-push fallback: refresh the packed S-slot mirror rows
                # and push the whole slot arrays (the proven no-recompile
                # path; 0.77 GB/layer host mirror instead of 3.19 GB).
                self.slot13_host[slot] = w13_row
                self.slot2_host[slot] = w2_row
                if self.store.has_scales:
                    self.slot13_scale_host[slot] = s13_row
                    self.slot2_scale_host[slot] = s2_row
                self.slot_w13 = jax.device_put(
                    self._unpack_weight_rows(self.slot13_host,
                                             self._w13_packed),
                    self.dev_w13_sharding)
                self.slot_w2 = jax.device_put(
                    self._unpack_weight_rows(self.slot2_host,
                                             self._w2_packed),
                    self.dev_w2_sharding)
                if self.store.has_scales:
                    self.slot_w13_scale = jax.device_put(
                        self.slot13_scale_host, self.dev_w13_scale_sharding)
                    self.slot_w2_scale = jax.device_put(
                        self.slot2_scale_host, self.dev_w2_scale_sharding)
                self.slot_w13.block_until_ready()
                self.slot_w2.block_until_ready()
                if self.store.has_scales:
                    self.slot_w13_scale.block_until_ready()
                    self.slot_w2_scale.block_until_ready()
        else:
            self.slot13_host[slot] = self._unpack_weight_rows(
                self.w13_host[expert_id:expert_id + 1], self._w13_packed)[0]
            self.slot2_host[slot] = self._unpack_weight_rows(
                self.w2_host[expert_id:expert_id + 1], self._w2_packed)[0]
            if self.w13_scale_host is not None:
                self.slot13_scale_host[slot] = self.w13_scale_host[expert_id]
                self.slot2_scale_host[slot] = self.w2_scale_host[expert_id]
            self.slot_w13 = jax.device_put(self.slot13_host,
                                           self.dev_w13_sharding)
            self.slot_w2 = jax.device_put(self.slot2_host, self.dev_w2_sharding)
            if self.w13_scale_host is not None:
                self.slot_w13_scale = jax.device_put(
                    self.slot13_scale_host, self.dev_w13_scale_sharding)
                self.slot_w2_scale = jax.device_put(
                    self.slot2_scale_host, self.dev_w2_scale_sharding)
            self.slot_w13.block_until_ready()
            self.slot_w2.block_until_ready()
            if self.w13_scale_host is not None:
                self.slot_w13_scale.block_until_ready()
                self.slot_w2_scale.block_until_ready()
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
        """Current device slot contents: (w13, w2, w13_scale, w2_scale).

        Scale entries are None on unquantized banks. Consumers that feed the
        GMM kernel should prefer the individual slot_w13 / slot_w2 /
        slot_w13_scale / slot_w2_scale attributes; this tuple is the
        aggregate accessor.
        """
        return self.slot_w13, self.slot_w2, self.slot_w13_scale, self.slot_w2_scale

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


def _host_array(layer_name: str, field: str, value: jax.Array) -> np.ndarray:
    """Materialize a host bank, optionally packed or disk-backed.

    Packed FP4 banks keep the complete processed expert weights in anonymous
    memory at half their NumPy footprint. Disk-backed mode remains available
    for hosts without enough RAM, but is intentionally not used on Kaggle's
    slow overlay filesystem.
    """
    array = np.asarray(jax.device_get(value))
    if (envs.MOE_EXPERT_OFFLOAD_PACKED_HOST
            and getattr(array.dtype, "name", "") == "float4_e2m1fn"):
        raw = array.view(np.uint8)
        if raw.shape[-1] % 2:
            raise ValueError(
                f"cannot nibble-pack odd FP4 axis for {layer_name}.{field}: "
                f"{raw.shape}")
        array = ((raw[..., 0::2] & 0x0F)
                 | ((raw[..., 1::2] & 0x0F) << 4)).copy()
    if not envs.MOE_EXPERT_OFFLOAD_DISK_BACKED:
        return array

    storage_dir = Path(envs.MOE_EXPERT_OFFLOAD_STORAGE_DIR)
    storage_dir.mkdir(parents=True, exist_ok=True)
    safe_layer = layer_name.replace("/", "_").replace(".", "_")
    path = storage_dir / f"{safe_layer}.{field}.npy"
    mmap = np.lib.format.open_memmap(
        path, mode="w+", dtype=array.dtype, shape=array.shape)
    mmap[...] = array
    mmap.flush()
    del array
    return mmap


# ---------------------------------------------------------------------------
# Design D — canonical file-backed TPU-ready expert store
# ---------------------------------------------------------------------------
#
# One file per offloaded layer, one fixed-size record per expert. A record is
# the byte-identical concatenation of one expert's processed (GMM-layout)
# weight rows and fp32 block-scale rows — exactly the slices the anonymous
# bank (PACKED_HOST mode) holds in RAM, so W1 bit-identity is a memcmp. The
# file is the persistent representation of the layer's experts: reclaimable
# page cache (storage class "file"), never anonymous RAM.
#
# File layout (little-endian, records 4 KiB aligned):
#   [0:12288]                 header
#   [12288 + e * record]      record e  (n_experts records)
#
# Header (12288 bytes = 3 x 4 KiB):
#   [0:8]     magic b"DSV4EPRS"
#   [8:12]    version u32 (= 1)
#   [12:16]   layer_id u32
#   [16:20]   n_experts u32
#   [20]      weight dtype tag u8: 0 = nibble-packed float4_e2m1fn,
#             1 = fp32, 2 = bfloat16 (raw rows, unquantized banks)
#   [21:25]   ndim u8 x4: w13, w2, s13, s2 (0 = scale absent)
#   [25:89]   shape u32 x16: w13(4), w2(4), s13(4), s2(4), 0-padded
#   [89:129]  w13/w2/s13/s2/record/data bytes u64 x6
#   [137:...] per-record sha256 (32 B each), zero-padded to the end
#
# DeepSeek-V4-Flash exact byte counts (block-256 MXFP4, hidden 4096,
# moe_intermediate 2048, 256 experts):
#   record: 8,388,608 (w13 packed [4096,2048] u8)
#          + 4,194,304 (w2 packed [4096,1024] u8)
#          + 262,144 (w13 scales [16,1,4096] f32)
#          + 131,072 (w2 scales [8,1,4096] f32)
#          = 12,976,128 B (12.375 MiB)
#   layer file data: 256 * 12,976,128 = 3,321,888,768 B (3.094 GiB)
#   40 offloaded layers: 125.9 GiB of file-backed records total.
#
# Writes are verify-then-publish: <path>.tmp is filled, fsync'd, and renamed
# over the final name, so a reader never observes a partial layer.

_STORE_MAGIC = b"DSV4EPRS"
_STORE_VERSION = 1
_STORE_HEADER_BYTES = 12288
_STORE_SHA_OFF = 137
_STORE_MAX_RECORDS = (_STORE_HEADER_BYTES - _STORE_SHA_OFF) // 32

_STORE_W_FLOAT4_PACKED = 0
_STORE_W_FP32 = 1
_STORE_W_BF16 = 2


class ExpertStoreError(RuntimeError):
    """Loud store failure: missing, truncated, or sha-mismatched record."""


def store_layer_path(store_dir, layer_name: str):
    """Canonical store file for a layer: <dir>/layer_<L:03d>.rec."""
    layer_id = parse_layer_id(layer_name)
    return Path(store_dir) / f"layer_{layer_id:03d}.rec"


def parse_layer_id(layer_name: str) -> int:
    """Extract the integer layer index from a vLLM FusedMoE prefix."""
    match = re.search(r"\.layers\.(\d+)\.", layer_name)
    if match is None:
        raise ValueError(
            f"[expert-store] cannot parse layer index from {layer_name!r}")
    return int(match.group(1))


def store_enabled() -> bool:
    """Design D master switch (requires offload itself to be enabled)."""
    return offload_enabled() and bool(envs.MOE_EXPERT_OFFLOAD_STORE)


def push_mode() -> str:
    """Store-backed miss-refresh mode: 'scatter' | 'full' (validated in envs)."""
    mode = str(envs.MOE_EXPERT_OFFLOAD_PUSH_MODE).lower()
    if mode not in ("scatter", "full"):
        raise ValueError(
            f"[expert-store] MOE_EXPERT_OFFLOAD_PUSH_MODE must be "
            f"'scatter' or 'full', got {mode!r}")
    return mode


def _weight_dtype_tag(array: np.ndarray) -> int:
    name = getattr(array.dtype, "name", "")
    if name == "float4_e2m1fn":
        return _STORE_W_FLOAT4_PACKED
    if name == "float32":
        return _STORE_W_FP32
    if name == "bfloat16":
        return _STORE_W_BF16
    raise ValueError(
        f"[expert-store] unsupported weight dtype for store record: {name!r}")


def _shape_u32s(shape) -> bytes:
    out = bytearray(16)
    for i, dim in enumerate(shape or ()):
        if i >= 4 or int(dim) < 0 or int(dim) > 0xFFFFFFFF:
            raise ValueError(f"[expert-store] unsupported shape {shape!r}")
        struct.pack_into("<I", out, i * 4, int(dim))
    return bytes(out)


def _build_store_header(layer_id: int, n_experts: int, w_dtype_tag: int,
                        ndims, shapes, w13_bytes, w2_bytes, s13_bytes,
                        s2_bytes, record_bytes, data_bytes, shas) -> bytes:
    buf = bytearray(_STORE_HEADER_BYTES)
    buf[0:8] = _STORE_MAGIC
    struct.pack_into("<I", buf, 8, _STORE_VERSION)
    struct.pack_into("<I", buf, 12, layer_id)
    struct.pack_into("<I", buf, 16, n_experts)
    buf[20] = w_dtype_tag
    for i, ndim in enumerate(ndims):
        buf[21 + i] = ndim
    buf[25:89] = b"".join(_shape_u32s(shape) for shape in shapes)
    struct.pack_into("<Q", buf, 89, w13_bytes)
    struct.pack_into("<Q", buf, 97, w2_bytes)
    struct.pack_into("<Q", buf, 105, s13_bytes)
    struct.pack_into("<Q", buf, 113, s2_bytes)
    struct.pack_into("<Q", buf, 121, record_bytes)
    struct.pack_into("<Q", buf, 129, data_bytes)
    buf[_STORE_SHA_OFF:_STORE_SHA_OFF + 32 * n_experts] = b"".join(shas)
    return bytes(buf)


def write_expert_store(path, layer_id: int, w13: np.ndarray, w2: np.ndarray,
                       w13_scale: np.ndarray | None = None,
                       w2_scale: np.ndarray | None = None) -> int:
    """Write one layer's processed records to the canonical store.

    ``w13``/``w2`` are GMM-layout processed weight arrays: float4_e2m1fn
    (itemsize 1; nibble-packed into 2 codes/byte here, the exact encoding of
    the PACKED_HOST bank rows) or raw fp32/bf16 for unquantized banks.
    ``w13_scale``/``w2_scale`` are the 4-D fp32 block scales (or None for
    both). Verify-then-publish: the file is filled at ``<path>.tmp``, fsync'd
    with its per-record sha256 table, then renamed. Returns record_bytes.
    """
    if (w13_scale is None) != (w2_scale is None):
        raise ValueError(
            f"[expert-store] layer {layer_id}: scales must be provided "
            "together (both None for unquantized)")
    w_dtype_tag = _weight_dtype_tag(w13)
    if _weight_dtype_tag(w2) != w_dtype_tag:
        raise ValueError(
            f"[expert-store] layer {layer_id}: w13/w2 weight dtype mismatch "
            f"({_weight_dtype_tag(w13)} vs {_weight_dtype_tag(w2)})")
    if w13.shape[0] != w2.shape[0]:
        raise ValueError(
            f"[expert-store] layer {layer_id}: w13/w2 expert count mismatch "
            f"({w13.shape[0]} vs {w2.shape[0]})")
    n_experts = int(w13.shape[0])
    if n_experts > _STORE_MAX_RECORDS:
        raise ValueError(
            f"[expert-store] layer {layer_id}: {n_experts} experts exceed "
            f"the {_STORE_MAX_RECORDS} per-file sha table")
    if w13.dtype.name == "float4_e2m1fn" and w13.shape[-1] % 2:
        raise ValueError(
            f"[expert-store] layer {layer_id}: cannot nibble-pack odd FP4 "
            f"axis {w13.shape}")
    if w13_scale is not None:
        if (w13_scale.shape[0] != n_experts
                or w2_scale.shape[0] != n_experts):
            raise ValueError(
                f"[expert-store] layer {layer_id}: scale expert count "
                "mismatch")
        if (w13_scale.dtype != np.float32
                or w2_scale.dtype != np.float32):
            raise ValueError(
                f"[expert-store] layer {layer_id}: block scales must be "
                f"fp32, got {w13_scale.dtype} / {w2_scale.dtype}")

    pack_float4 = w_dtype_tag == _STORE_W_FLOAT4_PACKED
    if pack_float4:
        w13_store = ((w13.view(np.uint8)[..., 0::2] & 0x0F)
                     | ((w13.view(np.uint8)[..., 1::2] & 0x0F) << 4)).copy()
        w2_store = ((w2.view(np.uint8)[..., 0::2] & 0x0F)
                    | ((w2.view(np.uint8)[..., 1::2] & 0x0F) << 4)).copy()
    else:
        w13_store = np.ascontiguousarray(w13)
        w2_store = np.ascontiguousarray(w2)
    s13_store = (None if w13_scale is None
                 else np.ascontiguousarray(w13_scale, dtype=np.float32))
    s2_store = (None if w2_scale is None
                else np.ascontiguousarray(w2_scale, dtype=np.float32))

    w13_bytes = int(w13_store[0].nbytes)
    w2_bytes = int(w2_store[0].nbytes)
    s13_bytes = 0 if s13_store is None else int(s13_store[0].nbytes)
    s2_bytes = 0 if s2_store is None else int(s2_store[0].nbytes)
    record_bytes = w13_bytes + w2_bytes + s13_bytes + s2_bytes
    data_bytes = record_bytes * n_experts

    tmp_path = Path(str(path) + ".tmp")
    # O_RDWR (not O_WRONLY): this kernel refuses ACCESS_WRITE mmaps through
    # an O_WRONLY fd with EACCES (measured on the box; O_RDWR works).
    fd = os.open(tmp_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, _STORE_HEADER_BYTES + data_bytes)
        mm = _mmap.mmap(fd, _STORE_HEADER_BYTES + data_bytes)
        shas = []
        base = _STORE_HEADER_BYTES
        for e in range(n_experts):
            off = base + e * record_bytes
            mm[off:off + w13_bytes] = w13_store[e].tobytes()
            off += w13_bytes
            mm[off:off + w2_bytes] = w2_store[e].tobytes()
            off += w2_bytes
            if s13_store is not None:
                mm[off:off + s13_bytes] = s13_store[e].tobytes()
                off += s13_bytes
            if s2_store is not None:
                mm[off:off + s2_bytes] = s2_store[e].tobytes()
            digest = hashlib.sha256(
                bytes(mm[base + e * record_bytes:
                          base + (e + 1) * record_bytes]))
            shas.append(digest.digest())
        header = _build_store_header(
            layer_id, n_experts, w_dtype_tag,
            # ROW ndims/shapes: the header describes ONE record (one
            # expert's rows), so the expert axis 0 is stripped. The reader
            # reshapes a single record's bytes against these.
            (w13_store.ndim - 1, w2_store.ndim - 1,
             0 if s13_store is None else s13_store.ndim - 1,
             0 if s2_store is None else s2_store.ndim - 1),
            (w13_store.shape[1:], w2_store.shape[1:],
             () if s13_store is None else s13_store.shape[1:],
             () if s2_store is None else s2_store.shape[1:]),
            w13_bytes, w2_bytes, s13_bytes, s2_bytes,
            record_bytes, data_bytes, shas)
        mm[0:_STORE_HEADER_BYTES] = header
        mm.flush()
        os.fsync(fd)
        mm.close()
    finally:
        os.close(fd)
    os.replace(tmp_path, path)
    return record_bytes


class _LayerStore:
    """Read-only mmap'd expert store for one layer, verified at open.

    Every record is sha256-checked against the header table at open (loud
    ExpertStoreError on any mismatch) and, when ``verify_read`` is set,
    again at fetch so a torn page or overlay fault never reaches a slot
    silently. Optional bounded LRU hot cache of raw record bytes
    (MOE_EXPERT_OFFLOAD_HOT_CACHE_GIB; 0 = off, the default).
    """

    def __init__(self, path, expected_layer_id: int,
                 hot_cache_bytes: int = 0, verify_read: bool = True):
        self.path = Path(path)
        try:
            size = self.path.stat().st_size
        except OSError as exc:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: file missing or unreadable: "
                f"{exc}") from exc
        with open(self.path, "rb") as f:
            header = f.read(_STORE_HEADER_BYTES)
        if len(header) < _STORE_HEADER_BYTES:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: truncated header "
                f"({len(header)} < {_STORE_HEADER_BYTES} bytes)")
        if header[0:8] != _STORE_MAGIC:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: bad magic {header[0:8]!r}")
        (version, layer_id, n_experts) = struct.unpack_from("<III",
                                                             header, 8)
        if version != _STORE_VERSION:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: unsupported version {version}")
        if layer_id != expected_layer_id:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: layer id {layer_id} != "
                f"expected {expected_layer_id}")
        w_dtype_tag = header[20]
        if w_dtype_tag not in (_STORE_W_FLOAT4_PACKED, _STORE_W_FP32,
                               _STORE_W_BF16):
            raise ExpertStoreError(
                f"[expert-store] {self.path}: bad weight dtype tag "
                f"{w_dtype_tag}")
        (w13_bytes, w2_bytes, s13_bytes, s2_bytes, record_bytes,
         data_bytes) = struct.unpack_from("<6Q", header, 89)
        if record_bytes != w13_bytes + w2_bytes + s13_bytes + s2_bytes:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: record_bytes "
                f"{record_bytes} != sum of parts")
        if data_bytes != record_bytes * n_experts:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: data_bytes {data_bytes} != "
                f"{n_experts} * {record_bytes}")
        if size != _STORE_HEADER_BYTES + data_bytes:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: file size {size} != header + "
                f"data ({_STORE_HEADER_BYTES + data_bytes})")
        if w13_bytes == 0 or w2_bytes == 0 or n_experts == 0:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: degenerate layout "
                f"(n={n_experts}, w13={w13_bytes}, w2={w2_bytes})")

        self.n_experts = n_experts
        self.record_bytes = record_bytes
        self.w13_bytes = w13_bytes
        self.w2_bytes = w2_bytes
        self.s13_bytes = s13_bytes
        self.s2_bytes = s2_bytes
        # Scale rows must be present together (write-time invariant).
        if (s13_bytes > 0) != (s2_bytes > 0):
            raise ExpertStoreError(
                f"[expert-store] {self.path}: scale rows must be present "
                "together")
        self.has_scales = s13_bytes > 0 and s2_bytes > 0
        # Row shapes (stored form: float4 rows are nibble-packed, last axis
        # halved; scales are the raw 4-D-processed row shapes).
        shapes = struct.unpack_from("<16I", header, 25)
        self.w13_ndim = header[21]
        self.w2_ndim = header[22]
        self.s13_ndim = header[23]
        self.s2_ndim = header[24]
        self.w13_shape = tuple(
            shapes[i] for i in range(self.w13_ndim))
        self.w2_shape = tuple(shapes[4 + i] for i in range(self.w2_ndim))
        self.s13_shape = tuple(shapes[8 + i]
                               for i in range(self.s13_ndim))
        self.s2_shape = tuple(shapes[12 + i] for i in range(self.s2_ndim))
        if w_dtype_tag == _STORE_W_FLOAT4_PACKED:
            self.w13_row_dtype = np.uint8
            self.w2_row_dtype = np.uint8
            self._float4_dtype = np.dtype("float4_e2m1fn")
        elif w_dtype_tag == _STORE_W_FP32:
            self.w13_row_dtype = np.float32
            self.w2_row_dtype = np.float32
            self._float4_dtype = None
        else:
            self.w13_row_dtype = np.dtype("bfloat16")
            self.w2_row_dtype = np.dtype("bfloat16")
            self._float4_dtype = None
        # Cross-check stored bytes against the declared row shapes.
        def _row_nbytes(dtype, shape) -> int:
            itemsize = np.dtype(
                "uint8" if dtype is np.uint8 else dtype).itemsize
            n = 1
            for dim in shape:
                n *= dim
            return n * itemsize
        if (_row_nbytes(self.w13_row_dtype, self.w13_shape) != w13_bytes
                or _row_nbytes(self.w2_row_dtype, self.w2_shape) != w2_bytes):
            raise ExpertStoreError(
                f"[expert-store] {self.path}: row shape/byte mismatch "
                f"(w13 {self.w13_shape} vs {w13_bytes} B)")
        if self.has_scales:
            if (_row_nbytes(np.float32, self.s13_shape) != s13_bytes
                    or _row_nbytes(np.float32, self.s2_shape) != s2_bytes):
                raise ExpertStoreError(
                    f"[expert-store] {self.path}: scale shape/byte mismatch")
        self._sha_table = [
            header[_STORE_SHA_OFF + 32 * e:
                    _STORE_SHA_OFF + 32 * (e + 1)]
            for e in range(n_experts)
        ]
        # Open-time verification: hash EVERY record against the header table
        # before the store is handed to a bank (loud failure at registration,
        # not mid-batch). The file is page-cache-hot right after the
        # load-time write; on a cold remount this is one sequential read.
        # Per-read re-verification (verify_read) remains the runtime guard.
        with open(self.path, "rb") as f:
            for e in range(n_experts):
                f.seek(_STORE_HEADER_BYTES + e * record_bytes)
                digest = hashlib.sha256(
                    f.read(record_bytes)).digest()
                if digest != self._sha_table[e]:
                    raise ExpertStoreError(
                        f"[expert-store] {self.path}: sha256 mismatch for "
                        f"record expert={e} at open (expected "
                        f"{self._sha_table[e].hex()[:16]}..., got "
                        f"{digest.hex()[:16]}...) — corrupt or torn store "
                        "page; rebuild the layer store")
        # Hold the fd first: the mmap needs a LIVE descriptor, and a
        # throwaway open(...).fileno() closes on GC before the mmap maps.
        self._keep_fd = open(self.path, "rb")
        self._mm = _mmap.mmap(self._keep_fd.fileno(), 0,
                              access=_mmap.ACCESS_READ)
        self.verify_read = verify_read
        self._hot: "dict[int, bytes]" = ({} if hot_cache_bytes > 0 else None)
        self._hot_max = max(int(hot_cache_bytes), 0)
        self._hot_used = 0

    def close(self) -> None:
        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None
        if getattr(self, "_keep_fd", None) is not None:
            self._keep_fd.close()
            self._keep_fd = None

    def _record_range(self, e: int, verify: bool):
        if not 0 <= e < self.n_experts:
            raise ExpertStoreError(
                f"[expert-store] {self.path}: expert {e} out of range "
                f"[0, {self.n_experts})")
        off = _STORE_HEADER_BYTES + e * self.record_bytes
        raw = bytes(self._mm[off:off + self.record_bytes])
        if verify:
            digest = hashlib.sha256(raw).digest()
            if digest != self._sha_table[e]:
                raise ExpertStoreError(
                    f"[expert-store] {self.path}: sha256 mismatch for "
                    f"record expert={e} (expected "
                    f"{self._sha_table[e].hex()[:16]}..., got "
                    f"{digest.hex()[:16]}...) — corrupt or torn store "
                    "page; rebuild the layer store")
        return raw

    def read_record(self, e: int) -> tuple:
        """Fetch one expert's rows: (w13, w2, s13 or None, s2 or None).

        Rows are read-only views in STORED form (float4 weights nibble-
        packed; scales fp32). Callers that mutate must copy (the scatter
        unpack and the full-push mirror assignment both copy).
        """
        if self._hot is not None and e in self._hot:
            raw = self._hot.pop(e)
            self._hot[e] = raw  # refresh LRU recency (dict = insertion order)
        else:
            raw = self._record_range(e, verify=self.verify_read)
            if self._hot is not None:
                self._hot[e] = raw
                self._hot_used += len(raw)
                while self._hot_used > self._hot_max and len(self._hot) > 1:
                    oldest = next(iter(self._hot))
                    self._hot_used -= len(self._hot.pop(oldest))
        w13 = np.frombuffer(raw[:self.w13_bytes], dtype=self.w13_row_dtype)
        w13 = w13.reshape(self.w13_shape)
        w2 = np.frombuffer(raw[self.w13_bytes:
                               self.w13_bytes + self.w2_bytes],
                           dtype=self.w2_row_dtype)
        w2 = w2.reshape(self.w2_shape)
        pos = self.w13_bytes + self.w2_bytes
        if self.has_scales:
            s13 = np.frombuffer(raw[pos:pos + self.s13_bytes],
                                dtype=np.float32)
            s13 = s13.reshape(self.s13_shape)
            s2 = np.frombuffer(raw[pos + self.s13_bytes:], dtype=np.float32)
            s2 = s2.reshape(self.s2_shape)
        else:
            s13 = s2 = None
        return w13, w2, s13, s2

    def read_record_bytes(self, e: int) -> bytes:
        """Raw (verified) record bytes for the optional hot cache/tests."""
        if self._hot is not None and e in self._hot:
            return self._hot[e]
        raw = self._record_range(e, verify=self.verify_read)
        if self._hot is not None:
            self._hot[e] = raw
            self._hot_used += len(raw)
            while self._hot_used > self._hot_max and len(self._hot) > 1:
                oldest = next(iter(self._hot))
                del self._hot[oldest]
                self._hot_used -= self.record_bytes
        return raw


def open_expert_store(path, expected_layer_id: int) -> _LayerStore:
    """Open + verify a layer store; loud ExpertStoreError on any mismatch."""
    return _LayerStore(
        path,
        expected_layer_id,
        hot_cache_bytes=(max(int(envs.MOE_EXPERT_OFFLOAD_HOT_CACHE_GIB), 0)
                         * _BYTES_PER_GIB),
    )


def store_write_layer(layer_name: str, w13, w2, w13_scale=None,
                      w2_scale=None) -> Path:
    """Write one layer's processed records to the canonical store.

    ``w13``/``w2`` are the GMM-layout processed arrays exactly as handed to
    register_bank (float4_e2m1fn weights or unquantized fp32/bf16);
    ``w13_scale``/``w2_scale`` are the 4-D fp32 block scales (or None for
    both). The processed arrays are materialized to host NumPy, written
    verify-then-publish into <STORE_DIR>/layer_<L:03d>.rec, and the final
    path is returned. This runs once per layer during the load-time
    transform (already-paid ~23 min), pipelined with no wall regression.
    """
    layer_id = parse_layer_id(layer_name)
    w13_np = np.asarray(jax.device_get(w13))
    w2_np = np.asarray(jax.device_get(w2))
    s13_np = (None if w13_scale is None
              else np.asarray(jax.device_get(w13_scale)).astype(
                  np.float32, copy=False))
    s2_np = (None if w2_scale is None
             else np.asarray(jax.device_get(w2_scale)).astype(
                 np.float32, copy=False))
    path = store_layer_path(envs.MOE_EXPERT_OFFLOAD_STORE_DIR, layer_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    record_bytes = write_expert_store(path, layer_id, w13_np, w2_np, s13_np,
                                      s2_np)
    logger.info(
        "[expert-offload] store layer %s: %s (%d experts, %d B/record, "
        "%.2f GiB file, sha-verified, verify-then-publish)",
        layer_name, path, w13_np.shape[0], record_bytes,
        path.stat().st_size / _BYTES_PER_GIB)
    return path


def register_bank(layer_name: str, w13_host: jax.Array, w2_host: jax.Array,
                 dev_w13_sharding, dev_w2_sharding,
                 w13_scale_host: jax.Array | None = None,
                 w2_scale_host: jax.Array | None = None,
                 dev_w13_scale_sharding=None, dev_w2_scale_sharding=None,
                 layer=None, store_path: str | None = None
                 ) -> _LayerBank | None:
    """Register a layer's host bank; returns the bank or None if disabled.

    Args:
        layer_name: Registry key, the vLLM FusedMoE prefix
            "model.layers.{i}.ffn.experts" (== layer.layer_name at serve time).
        w13_host / w2_host: GMM_TP processed expert weights (packed fp4 or
            unquantized), [N, ...] layout. IGNORED (and must be the jax
            arrays the caller already built) when ``store_path`` is given.
        dev_w13_sharding / dev_w2_sharding: NamedSharding for the weights
            (2-D GMM_TP specs).
        w13_scale_host / w2_scale_host: fp32 block scales for quantized
            banks, 4-D GMM_TP processed layout ([N, 8, 1, 4096] /
            [N, 4, 1, 4096] for DeepSeek-V4 0731). Both None for
            unquantized banks.
        dev_w13_scale_sharding / dev_w2_scale_sharding: NamedSharding for the
            scales -- the 4-D kernel specs P(None,None,None,MLP_TENSOR) /
            P(None,MLP_TENSOR,None,None), NOT the 2-D weight specs.
        layer: The RoutedExperts layer being registered, used for the shared
            hash_routed guard. Hash-routed layers (hash_indices_table set,
            e.g. DeepSeek-V4 layers 0-2) are REFUSED here -- the single
            choke point covering both the unquantized and the MXFP4
            registration gates -- because banking them would feed S-column
            gating together with hash indices in expert-id space [0, N), an
            out-of-bounds take_along_axis in fused_moe_gmm.py.
        store_path: Design D -- path to this layer's canonical expert store
            file. When given, the store (opened, header- and sha-verified)
            backs the bank and the anonymous full-bank arrays are not
            retained; MOE_EXPERT_OFFLOAD_PUSH_MODE selects the scatter /
            full push behavior.

    Returns:
        The registered bank, or None when offload is disabled or the layer
        is hash-routed (callers must fall through to the full shard path).
    """
    if not offload_enabled():
        return None
    if layer is not None and getattr(layer, "hash_indices_table", None) is not None:
        logger.warning_once(
            "[expert-offload] layer %s: hash-routed (hash_indices_table set); "
            "refusing to bank -- layer stays resident on the full-bank path.",
            layer_name)
        return None
    if store_path is not None:
        store = open_expert_store(store_path, parse_layer_id(layer_name))
        bank = _LayerBank(
            layer_name, None, None,
            dev_w13_sharding,
            dev_w2_sharding,
            dev_w13_scale_sharding=dev_w13_scale_sharding,
            dev_w2_scale_sharding=dev_w2_scale_sharding,
            store=store,
            push_mode=push_mode(),
        )
    elif envs.MOE_EXPERT_OFFLOAD_RAW_JIT:
        raise NotImplementedError(
            "[expert-offload] MOE_EXPERT_OFFLOAD_RAW_JIT=1 selects the "
            "just-in-time per-expert transform (Design C), which is not "
            "implemented on this branch; use the store path "
            "(MOE_EXPERT_OFFLOAD_STORE=1, default) or the anonymous bank "
            "(MOE_EXPERT_OFFLOAD_STORE=0).")
    else:
        bank = _LayerBank(
            layer_name,
            _host_array(layer_name, "w13", w13_host),
            _host_array(layer_name, "w2", w2_host),
            dev_w13_sharding,
            dev_w2_sharding,
            None if w13_scale_host is None else _host_array(
                layer_name, "w13_scale", w13_scale_host),
            None if w2_scale_host is None else _host_array(
                layer_name, "w2_scale", w2_scale_host),
            dev_w13_scale_sharding,
            dev_w2_scale_sharding,
        )
    _BANKS[layer_name] = bank
    return bank