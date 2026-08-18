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

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

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
                 dev_w13_scale_sharding=None, dev_w2_scale_sharding=None):
        self.layer_name = layer_name
        self.w13_host = w13_host            # processed host mirror [N, ...]
        self.w2_host = w2_host              # processed host mirror [N, ...]
        # JAX float4 uses one byte per value in host NumPy storage, although
        # only the low nibble is meaningful. Pack two codes per byte to keep
        # all 40 processed banks resident without an overlay/NFS write path.
        self._w13_packed = (w13_host.dtype == np.uint8
                            and envs.MOE_EXPERT_OFFLOAD_PACKED_HOST)
        self._w2_packed = (w2_host.dtype == np.uint8
                           and envs.MOE_EXPERT_OFFLOAD_PACKED_HOST)
        self._float4_dtype = (np.dtype("float4_e2m1fn")
                              if self._w13_packed or self._w2_packed else None)
        self.dev_w13_sharding = dev_w13_sharding   # GMM_TP: P(None, None, MLP_TENSOR)
        self.dev_w2_sharding = dev_w2_sharding     # GMM_TP: P(None, MLP_TENSOR, None)
        # MXFP4/FP4 block scales (fp32, GMM_TP processed 4-D layout) mirrored
        # on host; None for unquantized banks.
        self.w13_scale_host = w13_scale_host  # host mirror [N, 16, 1, 4096] or None
        self.w2_scale_host = w2_scale_host    # host mirror [N, 8, 1, 4096] or None
        # 4-D kernel scale shardings: P(None,None,None,MLP_TENSOR) /
        # P(None,MLP_TENSOR,None,None) -- NOT the 2-D weight shardings above.
        self.dev_w13_scale_sharding = dev_w13_scale_sharding
        self.dev_w2_scale_sharding = dev_w2_scale_sharding
        if (w13_scale_host is None) != (w2_scale_host is None):
            raise ValueError(
                f"[expert-offload] layer {layer_name}: w13/w2 scale hosts "
                "must be provided together (both None for unquantized).")
        if (w13_scale_host is not None
                and (dev_w13_scale_sharding is None
                     or dev_w2_scale_sharding is None)):
            raise ValueError(
                f"[expert-offload] layer {layer_name}: scale shardings "
                "required when scale hosts are provided.")
        self.slots = slot_count()           # MOE_EXPERT_OFFLOAD_SLOTS (16 default)
        self._allocate_initial_slots()
        host_bytes = (self.w13_host.nbytes + self.w2_host.nbytes
                      + (0 if self.w13_scale_host is None
                         else self.w13_scale_host.nbytes
                         + self.w2_scale_host.nbytes))
        logger.info(
            "[expert-offload] layer %s: host bank %d experts, %d device slots "
            "(%.2f GB host, scales %s)",
            self.layer_name, self.w13_host.shape[0], self.slots,
            host_bytes / 1e9,
            "yes" if self.w13_scale_host is not None else "no")

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
        slots hold experts 1..S-1. S*PER_EXP math: S=16 -> 2*16*10.486MB =
        0.3355 GB ("slots 0.34 GB" in the stage-4 falsification output).
        Quantized banks device_put the fp32 block-scale slots with their 4-D
        kernel shardings alongside the packed weights.
        """
        S = self.slots
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
        pair atomically).
        """
        self.slot13_host[slot] = self._unpack_weight_rows(
            self.w13_host[expert_id:expert_id + 1], self._w13_packed)[0]
        self.slot2_host[slot] = self._unpack_weight_rows(
            self.w2_host[expert_id:expert_id + 1], self._w2_packed)[0]
        if self.w13_scale_host is not None:
            self.slot13_scale_host[slot] = self.w13_scale_host[expert_id]
            self.slot2_scale_host[slot] = self.w2_scale_host[expert_id]
        self.slot_w13 = jax.device_put(self.slot13_host, self.dev_w13_sharding)
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


def register_bank(layer_name: str, w13_host: jax.Array, w2_host: jax.Array,
                  dev_w13_sharding, dev_w2_sharding,
                  w13_scale_host: jax.Array | None = None,
                  w2_scale_host: jax.Array | None = None,
                  dev_w13_scale_sharding=None, dev_w2_scale_sharding=None,
                  layer=None) -> _LayerBank | None:
    """Register a layer's host bank; returns the bank or None if disabled.

    Args:
        layer_name: Registry key, the vLLM FusedMoE prefix
            "model.layers.{i}.ffn.experts" (== layer.layer_name at serve time).
        w13_host / w2_host: GMM_TP processed expert weights (packed fp4 or
            unquantized), [N, ...] layout.
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