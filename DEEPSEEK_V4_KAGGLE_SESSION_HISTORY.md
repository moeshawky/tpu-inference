# DeepSeek-V4-Flash-0731 on Kaggle TPU v5e-8
## Complete session history and restart handoff

**Recorded:** 2026-08-18 15:50 UTC  
**Platform:** Kaggle TPU v5e-8, single host, 8 chips  
**Repository:** `/kaggle/working/tpu-inference`  
**Model:** `/kaggle/input/models/moeshawky/deepseekv4-flash-ga-0731-dspark-abliterated/transformers/deepseekv4-flash-ga-0731-hf/1/deepseekv4flash`  
**Status:** DeepSeek serving is not proven. Current server is stopped cleanly. The next run should start from the packed-host/direct-I/O code state described below.

> The requested `documentarian` skill was not available in the installed skill catalog or repository. This file is the structured replacement handoff.

---

## 1. Operator objective

Serve DeepSeek-V4-Flash-0731 through the fork's TPU vLLM path with:

- `MOE_EXPERT_OFFLOAD=1`
- 64 device expert slots per offloaded MoE layer
- host-backed full expert banks
- MXFP4/FP4 weights with block scales
- TPU v5e-8, TP=8
- DeepSeek V4 native FP4 path
- eventual proof via a real generated token, not only device attachment or weight loading

The immediate operational constraint is that a previous uncontrolled host-memory peak restarted the Kaggle notebook. The new implementation must load successfully without causing another notebook restart, while not rejecting the model merely because Kaggle's NFS page cache is charged in `memory.current`.

---

## 2. Canonical machine facts

### TPU and software

- TPU: 8 x v5e chips, one host.
- HBM per chip reported by the worker: 15.75 GiB usable.
- `libtpu`: 0.0.17.
- JAX sees 8 TPU devices with the full topology combination:

```bash
PJRT_DEVICE=TPU
TPU_SKIP_MDS_QUERY=1
TPU_ACCELERATOR_TYPE=v5litepod-8
TPU_PROCESS_BOUNDS=1,1,1
TPU_WORKER_ID=0
TPU_WORKER_HOSTNAMES=localhost
TPU_PROCESS_ADDRESSES=local
TPU_CHIPS_PER_HOST_BOUNDS=2,4,1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
```

- The vLLM environment is isolated through `/usr/local/bin/kaggle-backend`.
- Do not replace system `torch`, `torch_xla`, `jax`, `jaxlib`, or `libtpu`.
- vLLM version in the isolated environment: 0.26.0.
- The notebook's Cell 9 supervisor is alive and must not be killed or restarted.
- Notebook identity is read from `/tmp/moe-kaggle-tpu/blocking-loop.json`; the current recorded kernel PID is 14. Never hardcode a PID in operational code.
- Do not touch `/tmp/moe-kaggle-tpu/` except to read the authoritative state file.

### Current clean machine state

Verified immediately before writing this handoff:

- No `vllm serve`, `EngineCore`, or `kaggle-backend run` server processes remain.
- All 8 `/dev/vfio/*` TPU chips report `PID N/A`.
- Notebook supervisor is active, heartbeat count 716, emergency worker idle.
- Physical memory exposed in `/proc/meminfo`: approximately 377.8 GiB usable total.
- Kaggle cgroup hard limit: `354334801920` bytes = **330.00 GiB**.
- The cgroup filesystem is mounted read-only; the limit cannot be raised from this process.
- No swap is available.
- Current cgroup after server shutdown:
  - raw `memory.current`: approximately 137.95 GiB
  - `memory.max`: 330.00 GiB
  - `file`: approximately 135.09 GiB
  - `shmem`: approximately 8.45 GiB
  - committed estimate (`current - file + shmem`): approximately 11.31 GiB
  - effective committed headroom: approximately 318.69 GiB
- `memory.events` records historical pressure:
  - `oom 86`
  - `oom_kill 1`
  - no new OOM event was recorded during the latest controlled runs.

The large current file charge is primarily cached Kaggle/NFS checkpoint data. It persists after the process exits because the Kaggle cgroup/NFS setup does not honor the attempted page-cache eviction reliably.

---

## 3. Repository and git boundary

### Committed checkpoint

Current HEAD:

```text
2e038833 Preserve DeepSeek MXFP4 offload checkpoint (incomplete)
```

This checkpoint intentionally records an incomplete/non-working implementation. It preserved the earlier MXFP4 processing, resharding, missing GMM-TP helper, and initial host-memory safety work after the notebook crash.

Parent/origin context:

```text
efd0ec03 (origin/main) Extend expert offload to MXFP4/FP4 experts with block-scale slots
```

### Current uncommitted files

```text
tests/layers/vllm/test_expert_offload.py
tpu_inference/envs.py
tpu_inference/layers/vllm/expert_offload.py
tpu_inference/layers/vllm/quantization/mxfp4.py
tpu_inference/models/vllm/vllm_model_loader.py
```

The current working tree is intentionally uncommitted because the packed-host path has not yet completed a full TPU model load.

Do not discard these changes without first preserving them. Do not call this implementation working until a clean run reaches server readiness and produces a real response.

---

## 4. Chronological debugging history

### 4.1 Earlier validation

Qwen3.5 TP=8 validation succeeded on the fork. That established that the TPU topology, isolated environment, and general vLLM TPU path can work on this machine. It did not validate DeepSeek-V4 MXFP4 offload.

### 4.2 Initial DeepSeek MXFP4 failure chain

The original DeepSeek attempt failed in several distinct stages:

1. **TPU HLO temporary-buffer OOM**
   - MXFP4 MoE processing on TPU attempted approximately 42.23 GiB of HLO temporaries on a 15.75 GiB chip.
   - This was a real TPU HBM compile/allocation failure, not host RAM.
   - Fix: run the dequantize/requantize processing under `cpu_mesh_context()` and explicitly place all processing inputs on the CPU device.

2. **CPU/TPU device mismatch**
   - `t2j()` commits arrays to the TPU by default.
   - CPU processing then rejected TPU-resident arguments.
   - Fix: `jax.device_put(..., jax.devices("cpu")[0])` for weights, scales, and optional biases inside the CPU mesh context.

3. **CPU output versus TPU reshard mismatch**
   - Processed weights remained CPU-resident and `shard_moe_weights()` attempted a TPU `NamedSharding` reshard.
   - Fix: replicate processed results back to the TPU mesh with `NamedSharding(self.mesh, P())` before `shard_moe_weights()`.

4. **Indivisible sharding failure**
   - A scale tensor with shape `(256, 4, 1, 4096)` could not be sharded across the selected axis on 8 devices.
   - The scale sharding contracts were corrected to the 4-D kernel layouts.

5. **Missing offload sharding methods**
   - The MXFP4 method lacked `_gmm_tp_w13_sharding()` and `_gmm_tp_w2_sharding()`.
   - These were added to match the unquantized implementation and satisfy `register_bank()`.

After these fixes, the server reached actual MXFP4 bank registration.

### 4.3 First successful bank-registration run, then Kaggle notebook OOM

A detached run progressed through 33 expert banks, layers 3 through 35. Examples of the observed bank footprint:

```text
host bank: 256 experts, 64 device slots
approximately 6.54 GB host per layer
```

The process then died during layer 36 processing. The surviving evidence had no Python `MemoryError`, no JAX traceback, and no explicit EngineCore exception. The notebook UI reported that the notebook attempted to allocate more memory than available and restarted.

The original analysis used host-level `/proc` memory and saw approximately 83 GiB available, but the later cgroup inspection showed that the process was near a **330 GiB cgroup limit**, not a 396 GiB physical-RAM limit. The likely peak was:

- cumulative full host banks: approximately 6.54 GB x 33+ layers
- dense/resident model/runtime state
- next layer's CPU MXFP4 transient working set
- Kaggle/NFS file cache charged inside the same cgroup

This was a genuine Kaggle-specific memory failure. The notebook survived only after the failed server was stopped and its process group cleared.

### 4.4 First host-memory guard

The first safety patch added:

- cgroup v1/v2 memory readers
- `/proc/meminfo` fallback
- default estimated CPU working set: 48 GiB
- explicit reserve: 24 GiB
- per-layer admission before CPU MXFP4 processing
- `jax.effects_barrier()`
- deletion of intermediates
- `gc.collect()`

This successfully prevented a kernel OOM, but the policy was too conservative and used the raw cgroup `memory.current - current` value. On the next run it refused layer 3 with:

```text
available=58.58 GiB
estimated CPU working set=48.00 GiB
safety reserve=24.00 GiB
```

That number treated approximately 120+ GiB of reclaimable NFS file cache as if it were committed anonymous process memory. The refusal was safe but not a valid capacity conclusion.

### 4.5 Lazy loading and page-cache attempts

Several controlled runs established the following:

1. `--safetensors-load-strategy lazy` disables vLLM's background NFS auto-prefetch, but normal NFS reads still populate the file cache.
2. A project-level `safe_open` wrapper called `POSIX_FADV_DONTNEED` after shard close.
3. On this Kaggle NFS mount, `posix_fadvise(..., POSIX_FADV_DONTNEED)` returned success but did not materially reduce the cgroup's active-file charge.
4. The cgroup filesystem is read-only, so `/sys/fs/cgroup/memory.reclaim` cannot be used.
5. A global `/proc/sys/vm/drop_caches` operation was not used; it would be a machine-wide side effect and would not solve the underlying loader design.

### 4.6 Direct-I/O staging attempt and its bug

A probe established that:

```bash
dd if=<NFS shard> of=/dev/shm/... bs=1M iflag=direct
```

works on this mount. Reading a 1 GiB shard through direct I/O increased cgroup usage by only about 0.93 GiB while the staged file existed.

The loader was then patched to:

- stage one NFS shard at a time under `/dev/shm/tpu_inference_safetensors`
- pass the staged path through vLLM's safetensors iterator
- unlink the staged file after the shard

The first full run reached 48/48 shards, but cgroup `shmem` grew to approximately 133.92 GiB. Root cause: safetensors returns mmap-backed tensors, and vLLM retained those tensors as model parameter storage. Unlinking the staged file did not release the tmpfs pages while the mappings remained live.

That run was stopped before an OOM. The notebook remained alive.

### 4.7 Anonymous tensor clone fix

The direct-I/O iterator was changed from yielding mmap-backed tensors directly to:

```python
for name, tensor in original_iterator(...):
    detached_tensor = tensor.clone()
    del tensor
    yield name, detached_tensor
    del detached_tensor
```

This forces vLLM to retain anonymous CPU storage rather than a tmpfs-backed mapping.

A full one-shard validation passed:

- bytes yielded: `1059061760`
- staged file removed
- cgroup shmem before: approximately 8.45 GiB
- cgroup shmem after: approximately 8.45 GiB

This fix is verified for one complete shard, not yet for a full successful server boot.

### 4.8 Disk-backed bank attempt

To avoid keeping 6.54 GiB per bank in anonymous RAM, the host-bank registry was extended to write processed arrays as `.npy` memmaps under `/kaggle/temp/moe_expert_banks`.

The full run showed:

- direct-I/O shard loading safe after the clone fix
- first bank successfully materialized
- bank files totaling approximately 6.1 GiB
- committed memory stayed bounded
- however, overlay-backed memmap writes were extremely slow, approximately 10–20 MB/s in the observed phase

At that rate, writing all 40 banks was operationally impractical within a Kaggle session. The run was stopped cleanly. The disk-backed code remains in the working tree but is not the intended next-run mode.

### 4.9 Current packed-host direction

The processed JAX `float4_e2m1fn` dtype uses one byte per host NumPy value, but only the low nibble carries the FP4 code. A CPU probe showed bytes such as:

```text
float4_e2m1fn itemsize = 1
codes use the low 4 bits
```

The current code therefore adds optional host packing:

- two FP4 codes per byte in the full host bank
- unpack only the selected slot rows before `jax.device_put`
- scales remain normal FP32 arrays
- expected host footprint falls from ~6.54 GB/layer to roughly ~3.1 GB/layer
- no slow overlay write is needed when packed in-memory mode is enabled

The latest `/tmp/serve_deepseek.sh` sets:

```bash
MOE_EXPERT_OFFLOAD_DISK_BACKED=0
MOE_EXPERT_OFFLOAD_PACKED_HOST=1
```

This packed path has passed syntax checks and dtype-level probes, but it has **not** yet completed a full TPU bank registration or serving run. That is the exact restart boundary.

---

## 5. Current code changes by file

### `tpu_inference/layers/vllm/quantization/mxfp4.py`

Current uncommitted changes include:

- CPU mesh processing for the large MXFP4 dequantize/requantize path.
- Explicit CPU placement for t2j-produced inputs.
- Replication of processed CPU results onto the TPU mesh before normal sharding.
- `_gmm_tp_w13_sharding()` and `_gmm_tp_w2_sharding()` helpers.
- 4-D block-scale sharding helpers.
- Host-memory admission before processing offloaded layers.
- JAX barrier, explicit deletes, and garbage collection.
- Offload registration of processed FP4 weights plus scales.

### `tpu_inference/layers/vllm/expert_offload.py`

Current uncommitted changes include:

- cgroup memory readers and `memory.stat` parsing
- distinction between raw cgroup current, committed estimate, and reclaimable file cache
- memory admission telemetry
- disk-backed memmap support, retained as an optional mode
- packed FP4 host representation
- unpack-on-slot-load logic
- existing 64-slot host-bank routing behavior

Important: the packed representation must be verified on TPU. The first successful log should report a materially smaller host-bank footprint than the prior 6.54 GB.

### `tpu_inference/models/vllm/vllm_model_loader.py`

Current uncommitted changes include:

- safe-open page-cache hint wrapper
- one-shard-at-a-time direct-I/O staging through `/dev/shm`
- patching both `weight_utils.safetensors_weights_iterator` and the direct binding imported by `default_loader`
- cloning each yielded safetensors tensor before vLLM retains it

The full one-shard clone test passed and showed no shmem growth.

### `tpu_inference/envs.py`

Current flags include:

```text
MOE_EXPERT_OFFLOAD_HOST_MEMORY_GUARD
MOE_EXPERT_OFFLOAD_HOST_MEMORY_RESERVE_GIB
MOE_EXPERT_OFFLOAD_CPU_WORKING_SET_GIB
MOE_EXPERT_OFFLOAD_DISK_BACKED
MOE_EXPERT_OFFLOAD_PACKED_HOST
MOE_EXPERT_OFFLOAD_STORAGE_DIR
```

### `tests/layers/vllm/test_expert_offload.py`

The focused test file was updated during the earlier checkpoint work. `pytest` is unavailable in both the system Python and the protected vLLM environment, so the suite has not been executed. Syntax checks and dependency-free/isolated probes were used instead.

---

## 6. Current launch script

`/tmp/serve_deepseek.sh` is detached with `setsid`, uses the full TPU environment combination, and currently sets:

```bash
MOE_EXPERT_OFFLOAD=1
MOE_EXPERT_OFFLOAD_SLOTS=64
MOE_EXPERT_OFFLOAD_HOST_MEMORY_GUARD=1
MOE_EXPERT_OFFLOAD_HOST_MEMORY_RESERVE_GIB=12
MOE_EXPERT_OFFLOAD_CPU_WORKING_SET_GIB=16
MOE_EXPERT_OFFLOAD_DISK_BACKED=0
MOE_EXPERT_OFFLOAD_PACKED_HOST=1
MOE_EXPERT_OFFLOAD_STORAGE_DIR=/kaggle/temp/moe_expert_banks
```

It also sets:

```bash
--safetensors-load-strategy lazy
--tensor_parallel_size 8
--dtype bfloat16
--kv-cache-dtype fp8
--max-num-seqs 64
--gpu_memory_utilization 0.92
--additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}'
```

The script clears only its own temporary staging/bank directories and `/tmp/libtpu_lockfile`; it does not touch notebook supervisor state.

---

## 7. What is proven and what is not

### Proven

- TPU topology attaches with the full env-var combination.
- Qwen3.5 TP=8 fork validation succeeded.
- DeepSeek MXFP4 processing can run on CPU instead of overflowing TPU HBM.
- CPU-to-TPU reshard path is fixed far enough to reach offload registration.
- GMM-TP helper methods are present and reached by the offload path.
- A prior run registered 33 full in-memory banks before host-memory failure.
- Raw cgroup memory limit is 330 GiB.
- NFS page-cache eviction through `posix_fadvise` is ineffective here.
- `O_DIRECT` shard staging works.
- Direct-I/O staging without clone retains tmpfs pages; this was diagnosed.
- Direct-I/O staging with per-tensor clone passed a complete one-shard test.
- Disk-backed bank registration can write a complete first bank and stays memory-safe, but it is too slow on the observed overlay path.
- All latest runs were detached and safely stopped without killing/restarting the notebook.

### Not proven

- Packed in-memory FP4 bank registration on TPU.
- Full 40-bank DeepSeek load with packed banks.
- TPU HBM usage after packed banks are installed.
- First generated token.
- Uvicorn readiness.
- Numerical correctness of live routed slot swaps in the packed representation.
- Whether all 40 banks fit under the 330 GiB cgroup with the actual runtime's non-model allocations.
- Whether the cgroup's reclaimable-file accounting remains safe during all MXFP4 processing phases.

---

## 8. Clean restart procedure

### Before launch

1. Read this file completely.
2. Do not touch `/tmp/moe-kaggle-tpu/`.
3. Verify the notebook supervisor is active from `blocking-loop.json`.
4. Verify no server process remains.
5. Verify all `/dev/vfio/*` chips show `PID N/A`.
6. Preserve the current uncommitted files before any reset or checkout.
7. Run:

```bash
cd /kaggle/working/tpu-inference
python3 -m py_compile \
  tpu_inference/envs.py \
  tpu_inference/layers/vllm/expert_offload.py \
  tpu_inference/layers/vllm/quantization/mxfp4.py \
  tpu_inference/models/vllm/vllm_model_loader.py
bash -n /tmp/serve_deepseek.sh
git diff --check
```

### Packed-path smoke checks

Run through the protected environment:

- `np.dtype("float4_e2m1fn")` must work.
- `_host_array()` with `MOE_EXPERT_OFFLOAD_PACKED_HOST=1` must halve the last-axis storage.
- `_LayerBank._unpack_weight_rows()` must reconstruct the original FP4 codes.
- A small CPU bank should initialize slot arrays without dtype or shape errors.

### Launch

Use the existing detached script:

```bash
date +%s > /tmp/vllm-deepseek-start
bash /tmp/serve_deepseek.sh
```

Do not run a long pasted command. Do not attach log trailing in the same process group. The script is already `setsid`-detached.

### Monitor the correct metrics

Do not use raw `memory.current` alone. Record:

```bash
current=$(cat /sys/fs/cgroup/memory.current)
file=$(awk '$1=="file"{print $2}' /sys/fs/cgroup/memory.stat)
shmem=$(awk '$1=="shmem"{print $2}' /sys/fs/cgroup/memory.stat)
committed=$((current-file+shmem))
```

Track:

- raw cgroup current and headroom
- committed estimate and effective headroom
- anonymous RSS of EngineCore
- shmem, which must remain near the 8.45 GiB baseline during shard loading
- bank count and host-bank log size
- `memory.events` for any new `oom` or `oom_kill`
- exact log timestamps for every layer/bank milestone

### Success criteria

The run is successful only when all are true:

1. 48/48 checkpoint shards load.
2. Packed host bank registration proceeds through the intended offload layers.
3. No new cgroup OOM event occurs.
4. The notebook supervisor remains alive.
5. TPU HBM does not report OOM.
6. vLLM reports Uvicorn/application startup.
7. A real chat completion generates a sensible response.
8. TPU ownership/utilization is observed during generation.

### Safe abort rule

If committed headroom approaches the explicit reserve or a new allocation begins while raw cgroup headroom is near zero, stop the detached server process group before the notebook supervisor does. Determine the PGID from `ps`; do not use a stale hardcoded PID:

```bash
pgid=$(ps -o pgid= -p "$(cat /tmp/vllm-deepseek.pid)" | tr -d ' ')
kill -TERM -- "-$pgid"
```

Then verify:

- no vLLM/EngineCore process remains
- all TPU chips return to PID N/A
- notebook heartbeat continues
- no new `oom_kill` event was added

---

## 9. Exact restart boundary

The current state is **not a serving state**. The last full run was stopped during the slow disk-backed bank path after one bank. The latest code now has:

```text
anonymous cloned safetensors tensors: verified on one full shard
packed FP4 in-memory host bank: implemented but not full-TPU verified
```

The next action is therefore **not** to retry the old 48+24 GiB guard and **not** to wait for overlay memmap banks. The next action is:

1. run packed-host CPU smoke tests,
2. launch the current `setsid` script,
3. verify the first packed bank registration and its reduced host-byte log,
4. continue only if shmem stays flat and committed headroom remains large.

Do not reset the repository until the current five-file diff is preserved. Do not claim DeepSeek-V4 serving success until the generated-token test passes.
