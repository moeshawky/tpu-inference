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

"""§9B regression tests — align-mode dual-pool capacity invariant.

Qwen3.8/Qwen3.5 topology, prefix caching ON, mamba_cache_mode=align.
Invariant-based asserts only — NO exact block counts (dual-pool
accounting differs per deployment). See task.md §9B, §10 #5.
"""

import torch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                         KVCacheGroupSpec, MambaSpec)

from tpu_inference.core.hybrid_coordinator import (
    TPUDualBlockPool, TPUHybridKVCacheCoordinator, set_mamba_num_blocks)


def _make_qwen_kv_cache_config(
    num_attn_blocks: int = 2169,
    mamba_num_blocks: int = 129,
    block_size: int = 256,
) -> KVCacheConfig:
    """Qwen3.8/Qwen3.5 topology: one attention group + one Mamba group."""
    attn_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((3, 64), (8, 64, 16)),
        dtypes=(torch.bfloat16, torch.float32),
        block_size=block_size,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(kv_cache_spec=attn_spec, layer_names=["attn_0"]),
        KVCacheGroupSpec(kv_cache_spec=mamba_spec, layer_names=["mamba_0"]),
    ]
    cfg = KVCacheConfig(
        num_blocks=num_attn_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    set_mamba_num_blocks(mamba_num_blocks)
    return cfg


class TestAlignCapacityRegression:
    """Regression suite for the Sep-7 capacity-collapse failure.

    Failure mode: align guard skipped compact sizing, leaving attention
    collapsed to the uniform 494-block Mamba pool. Dual-pool split must
    restore attention capacity while keeping checkpoint IDs in bounds.
    """

    def test_align_does_not_collapse_attention_capacity(self):
        """Align mode must NOT collapse attention into uniform Mamba pool.

        Under prefix caching ON + align mode, attention blocks must be
        materially larger than the uniform-pool baseline of 494.
        Assert >1000 (§10 #5 bound), NOT exact 2169 — dual-pool
        accounting differs per deployment.
        """
        cfg = _make_qwen_kv_cache_config(num_attn_blocks=2169,
                                          mamba_num_blocks=129)
        coord = TPUHybridKVCacheCoordinator(
            kv_cache_config=cfg,
            max_model_len=1024,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            scheduler_block_size=256,
            hash_block_size=256,
        )
        attn_blocks = coord.attention_block_pool.num_gpu_blocks
        assert attn_blocks > 1000, (
            f"Attention pool collapsed to {attn_blocks} blocks "
            f"(uniform Mamba baseline is 494). "
            f"Dual-pool split failed — attention capacity is lost."
        )

    def test_mamba_checkpoint_block_ids_remain_valid(self):
        """Mamba checkpoint block IDs valid after dual-pool split.

        Checkpoint blocks allocated from the Mamba pool must have
        block_ids within [0, mamba_num_blocks). No OOB into the
        attention pool range. Dual-pool routing (free/touch) must also
        return each block to its originating pool.
        """
        mamba_num_blocks = 129
        set_mamba_num_blocks(mamba_num_blocks)
        cfg = _make_qwen_kv_cache_config(num_attn_blocks=2169,
                                          mamba_num_blocks=mamba_num_blocks)
        coord = TPUHybridKVCacheCoordinator(
            kv_cache_config=cfg,
            max_model_len=1024,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            scheduler_block_size=256,
            hash_block_size=256,
        )

        # Allocate checkpoint blocks from the Mamba pool.
        checkpoint_blocks = coord.mamba_block_pool.get_new_blocks(10)
        assert len(checkpoint_blocks) == 10

        # Every checkpoint block ID must be within mamba pool bounds.
        for blk in checkpoint_blocks:
            assert 0 <= blk.block_id < mamba_num_blocks, (
                f"Checkpoint block {blk.block_id} out of Mamba pool "
                f"bounds [0, {mamba_num_blocks}) — OOB indexing risk."
            )

        # Dual-pool routing: free_blocks must return blocks to the
        # mamba pool, not the attention pool (no cross-contamination).
        mamba_free_before = coord.mamba_block_pool.get_num_free_blocks()
        coord.block_pool.free_blocks(checkpoint_blocks)
        mamba_free_after = coord.mamba_block_pool.get_num_free_blocks()
        assert mamba_free_after == mamba_free_before + 10, (
            f"Mamba pool free count did not increase by 10 after freeing "
            f"checkpoint blocks: {mamba_free_before} -> {mamba_free_after}. "
            f"Cross-pool contamination detected."
        )
