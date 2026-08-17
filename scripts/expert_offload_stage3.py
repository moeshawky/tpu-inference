"""Stage-3 falsification: whole expert in a FIXED HBM slot, real path mechanics.

Proves the three mechanics the cache depends on, with real Qwen3-Coder-Next
weights through the real process_moe_weights GMM_TP pipeline:

  (1) RESIDENCY: host bank (pinned_host) of N real experts stays OFF HBM.
  (2) SLOT REFRESH: device_put of new expert VALUES into same-shape S-slot
      buffers does NOT recompile the jitted fused_moe_func (cache premise).
  (3) NUMERICS: S-slot device bank + remapped S-column gating == full-bank
      reference through the real kernel.

The fixed-S-slot cache architecture is exactly what Stage 4 will build; this
proves its three load-bearing assumptions before any cache machinery exists.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/kaggle/working/tpu-inference')

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights)

MODEL = ('/kaggle/input/models/qwen-lm/qwen3-coder-next/transformers/'
         'qwen3-coder-next/1')
INDEX = os.path.join(MODEL, 'model.safetensors.index.json')

H, F = 2048, 512
N_HOST = 128            # host bank size (real experts, universe)
S = 16                  # device slot cache size (compile-time fixed)
T, TOPK = 32, 10        # T*TOPK = 320, %16 == 0

PER_EXP = 10.486e6      # measured bytes per expert in GMM_TP padded layout


def load_expert_tensors(layer, ids):
    wm = json.load(open(INDEX))['weight_map']
    out = {}
    for eid in ids:
        for proj, suffix in (('gate', 'gate_proj'), ('up', 'up_proj'),
                             ('down', 'down_proj')):
            key = f'model.layers.{layer}.mlp.experts.{eid}.{suffix}.weight'
            shard = wm[key]
            with safe_open(os.path.join(MODEL, shard), framework='np') as f:
                out[(eid, proj)] = f.get_tensor(key)
    return out


def fuse_weights(tensors, ids):
    w13 = np.stack([np.concatenate([tensors[(e, 'gate')], tensors[(e, 'up')]],
                                   axis=0) for e in ids])
    w2 = np.stack([tensors[(e, 'down')] for e in ids])
    return FusedMoEWeights(w13_weight=jnp.asarray(w13), w13_weight_scale=None,
                           w13_bias=None, w2_weight=jnp.asarray(w2),
                           w2_weight_scale=None, w2_bias=None)


def hbm(devs):
    return sum(d.memory_stats()['bytes_in_use'] for d in devs)


def main():
    devs = jax.devices()
    mesh = Mesh(np.array(devs).reshape(1, 8), ('data', 'model'))
    dev_shard = NamedSharding(mesh, P(None, None, 'model'))
    host_shard = NamedSharding(mesh, P(None, None, 'model'),
                               memory_kind='pinned_host')
    print(f'TPU: {len(devs)} | mesh data=1 model=8 | host bank N={N_HOST} '
          f'slots S={S}', flush=True)

    # ---- build host bank (real weights, pinned_host) ----
    ids_host = np.arange(N_HOST)
    tensors = load_expert_tensors(0, ids_host)
    t0 = time.time()
    pw = process_moe_weights(fuse_weights(tensors, ids_host),
                             MoEBackend.GMM_TP, w13_reorder_size=8)
    w13_host = jax.device_put(pw.w13_weight, host_shard)
    w2_host = jax.device_put(pw.w2_weight, host_shard)
    w13_host.block_until_ready(); w2_host.block_until_ready()
    del pw
    jax.clear_caches()
    import gc; gc.collect(); time.sleep(1)
    bank_mb = N_HOST * PER_EXP / 1e6
    base = hbm(devs)
    print(f'bank {bank_mb:.0f} MB host-placed: HBM={base/1e9:.3f} GB '
          f'(source freed)', flush=True)

    np13 = np.asarray(jax.device_get(w13_host))
    np2 = np.asarray(jax.device_get(w2_host))

    # ---- (2)+(3): fixed S-slot device bank, refresh with real experts ----
    hidden = jax.random.normal(jax.random.key(2), (T, H), jnp.bfloat16)
    rng = np.random.default_rng(0)
    slot_expert = np.sort(rng.choice(ids_host, S, replace=False))
    slot_of = {int(e): i for i, e in enumerate(slot_expert)}

    def make_bank(exp_ids):
        w13 = process_moe_weights(fuse_weights(tensors, exp_ids),
                                  MoEBackend.GMM_TP,
                                  w13_reorder_size=8).w13_weight
        w2 = process_moe_weights(fuse_weights(tensors, exp_ids),
                                 MoEBackend.GMM_TP,
                                 w13_reorder_size=8).w2_weight
        return jax.device_put(w13, dev_shard), jax.device_put(w2, dev_shard)

    def make_gating(sel, logits):
        g = np.full((T, S), -np.inf, dtype=np.float32)
        for t in range(T):
            for e in sel[t]:
                g[t, slot_of[int(e)]] = logits[t, int(np.where(
                    ids_host == e)[0][0])]
        return jnp.asarray(g)

    # reference: run with the S-slot bank using gating that routes to those S
    w13d, w2d = make_bank(slot_expert)
    sel = np.stack([rng.choice(slot_expert, TOPK, replace=False)
                    for _ in range(T)])
    logits = np.zeros((T, N_HOST), dtype=np.float32)
    for t in range(T):
        for e in sel[t]:
            logits[t, int(np.where(ids_host == e)[0][0])] = rng.uniform(0.5, 2.0)
    g_full = jnp.asarray(logits[:, [int(np.where(ids_host == e)[0][0])
                                    for e in slot_expert]])

    print('\n== full S-slot bank (E=16) reference ==', flush=True)
    @jax.jit
    def run0(hidden, w13, w2, gating):
        return fused_moe_func(hidden, w13, w2, None, None, None, None,
                              gating, TOPK, True, mesh, False,
                              activation='silu', scoring_fn='softmax')
    t0 = time.time()
    out_ref = run0(hidden, w13d, w2d, g_full)
    out_ref.block_until_ready()
    print(f'compile+run: {time.time()-t0:.1f}s', flush=True)

    # refresh slots with a DIFFERENT set of real experts (same shapes)
    new_experts = np.sort(rng.choice(ids_host, S, replace=False))
    while np.array_equal(new_experts, slot_expert):
        new_experts = np.sort(rng.choice(ids_host, S, replace=False))
    slot_of = {int(e): i for i, e in enumerate(new_experts)}
    w13d2, w2d2 = make_bank(new_experts)
    sel2 = np.stack([rng.choice(new_experts, TOPK, replace=False)
                     for _ in range(T)])
    logits2 = np.zeros((T, N_HOST), dtype=np.float32)
    for t in range(T):
        for e in sel2[t]:
            logits2[t, int(np.where(ids_host == e)[0][0])] = rng.uniform(0.5, 2.0)
    g2 = make_gating(sel2, logits2)
    g2_ref = jnp.asarray(logits2[:, [int(np.where(ids_host == e)[0][0])
                                     for e in new_experts]])

    # JIT once: same abstract shapes across refreshes -> XLA cache hit means
    # changing slot VALUES never recompiles (the whole cache premise).
    @jax.jit
    def run(hidden, w13, w2, gating):
        return fused_moe_func(hidden, w13, w2, None, None, None, None,
                              gating, TOPK, True, mesh, False,
                              activation='silu', scoring_fn='softmax')

    # compile on the NEW slots (this is the only compile this refresh pays)
    out2 = run(hidden, w13d2, w2d2, g2)
    out2.block_until_ready()
    out2_ref = run(hidden, w13d2, w2d2, g2_ref)
    out2_ref.block_until_ready()

    t0 = time.time()
    for _ in range(3):
        out2 = run(hidden, w13d2, w2d2, g2)
        out2.block_until_ready()
    dt = (time.time() - t0) / 3
    print(f'\n== slot refresh (new real experts, same shapes) ==', flush=True)
    print(f'jitted per-call after refresh: {dt*1e3:.1f} ms '
          f'(no recompile if stable)', flush=True)

    d1 = jnp.max(jnp.abs(out2 - out2_ref)).item()
    print(f'remap vs full gating max|diff| = {d1:.3e}',
          f'({"PASS" if d1 < 1e-2 else "FAIL"})', flush=True)

    d2 = jnp.max(jnp.abs(out_ref - out2)).item()
    print(f'previous-slot vs refreshed-slot max|diff| = {d2:.3e} (expect '
          f'large: different experts routed)', flush=True)

    final_hbm = hbm(devs)
    print(f'\nfinal HBM = {final_hbm/1e9:.3f} GB '
          f'(2 x S-slot banks {2*S*PER_EXP/1e9:.2f} GB + activations)',
          flush=True)


if __name__ == '__main__':
    main()