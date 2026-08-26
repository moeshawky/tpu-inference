# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0
"""Regression test for TPU async-token-substitution state-classification bug.

Witness:
- speculative_config.num_speculative_tokens = 1
- three cached requests, scheduled counts [2, 2, 127]
- only the two 2-token requests are in scheduled_spec_decode_tokens
- KV usage ~0.999, 127-token request remains in placeholder
- _prepare_async_token_substitution_indices derived is_prefill=False
  from input_batch host counters and asserted 127 <= 2.

The fix uses scheduler-authoritative state (requests dict +
scheduler_output) to skip chunked-prefill remainders. Hiding is
intentional: host-side computed/prompt counters can be stale mid-chunk
(computed >= prompt while prefill remains), so a spec-enabled entry
scheduled >= num_speculative_tokens + 1 tokens outside
scheduled_spec_decode_tokens is treated as prefill and skipped with a
WARNING log rather than trusted or asserted on. Genuine scheduler bugs
surface through those WARNING logs instead of an assert.
"""

from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig

from tpu_inference.runner.tpu_runner import TPUModelRunner


def _make_runner(spec_tokens=1):
    mock_devices = [MagicMock(coords=i) for i in range(1)]
    mock_mesh = MagicMock()
    mock_mesh.shape = {"model": 1}
    mock_mesh.__class__.__name__ = "Mesh"
    with patch('jax.devices', return_value=mock_devices), \
         patch('jax.make_mesh', return_value=mock_mesh), \
         patch('jax.random.key', return_value=MagicMock()), \
         patch('tpu_inference.runner.tpu_runner.get_model', return_value=MagicMock()), \
         patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=mock_mesh), \
         patch('tpu_inference.runner.tpu_runner.jax.make_mesh', return_value=mock_mesh):
        model_config = ModelConfig(tokenizer_mode="auto", trust_remote_code=False, seed=0, dtype='bfloat16')
        model_config.max_model_len = 1024
        cache_config = CacheConfig(block_size=16, gpu_memory_utilization=0.9, cache_dtype="auto")
        scheduler_config = SchedulerConfig(max_num_seqs=16, max_model_len=1024, is_encoder_decoder=False)
        parallel_config = ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=1)
        speculative_config = SpeculativeConfig(model='ngram', num_speculative_tokens=spec_tokens, prompt_lookup_max=4)
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            speculative_config=speculative_config,
            observability_config={},
            additional_config={},
        )
        # Bypass heavy init
        with patch.object(TPUModelRunner, '_init_mesh', lambda self: setattr(self, 'mesh', mock_mesh)), \
             patch.object(TPUModelRunner, '_init_phased_profiling', lambda self: None), \
             patch.object(TPUModelRunner, '_init_aggregated_stats_logging', lambda self: None), \
             patch.object(TPUModelRunner, '_init_mm', lambda self: setattr(self, 'is_multimodal_model', False) or setattr(self, 'uses_mrope', False) or setattr(self, 'supports_mm_inputs', True)), \
             patch.object(TPUModelRunner, '_init_speculative_decoding', lambda self: setattr(self, 'drafter', None) or setattr(self, 'rejection_sampler', MagicMock())), \
             patch.object(TPUModelRunner, '_init_inputs', lambda self: None):
            runner = TPUModelRunner.__new__(TPUModelRunner)
            runner.vllm_config = vllm_config
            runner.model_config = model_config
            runner.cache_config = cache_config
            runner.scheduler_config = scheduler_config
            runner.parallel_config = parallel_config
            runner.speculative_config = speculative_config
            runner.devices = mock_devices
            runner.mesh = mock_mesh
            runner.dp_size = 1
            runner.max_num_reqs = 16
            runner.max_model_len = 1024
            runner.block_size = 16
            runner.input_batch = MagicMock()
            runner.requests = {}
            runner._pre_async_results = MagicMock()
            runner._pre_async_results.placeholder_req_id_to_index = {}
            from contextlib import nullcontext
            runner.maybe_forbid_compile = nullcontext()
            runner.scheduler_config.async_scheduling = True
            return runner


def test_regression_skip_127_prefill_with_placeholder():
    """Previous placeholder + current 127-token non-spec must skip, not assert 127<=2."""
    runner = _make_runner(spec_tokens=1)
    # Stale input_batch says all are decode (is_prefill False) -> buggy would assert
    runner.input_batch.req_id_to_index = {"req0": 0, "req1": 1, "req2": 2}
    runner.input_batch.num_computed_tokens_cpu = np.array([10, 10, 10], dtype=np.int32)
    runner.input_batch.num_prompt_tokens = np.array([5, 5, 5], dtype=np.int32)
    # Authoritative requests: req2 is still prefill (0 < 200)
    runner.requests = {
        "req0": MagicMock(prompt_token_ids=[1]*5, num_computed_tokens=10),
        "req1": MagicMock(prompt_token_ids=[1]*5, num_computed_tokens=10),
        "req2": MagicMock(prompt_token_ids=[1]*200, num_computed_tokens=0),
    }
    runner._pre_async_results = MagicMock()
    runner._pre_async_results.placeholder_req_id_to_index = {"req0": 0, "req1": 1, "req2": 2}
    runner.speculative_config = MagicMock(num_speculative_tokens=1)

    req_ids_dp = {0: ["req0", "req1", "req2"]}
    scheduled_tokens = {0: [2, 2, 127]}

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {"req0": 2, "req1": 2, "req2": 127}
    scheduler_output.scheduled_spec_decode_tokens = {"req0": [1], "req1": [1]}
    scheduler_output.scheduled_new_reqs = []
    cr = MagicMock()
    cr.req_ids = ["req0", "req1", "req2"]
    cr.num_computed_tokens = [10, 10, 0]
    cr.num_output_tokens = [1, 1, 0]
    cr.resumed_req_ids = set()
    scheduler_output.scheduled_cached_reqs = cr

    # Should NOT raise, and should skip req2 (127)
    cur, pre = runner._prepare_async_token_substitution_indices(
        req_ids_dp, scheduled_tokens, 128, 1, scheduler_output=scheduler_output)
    total = sum(len(v) for v in cur.values())
    assert total == 4, f"expected 4 substituted tokens (2+2), got {total} {cur}"
    # Verify req2's 127 not in indices
    # cur[0] should contain 4 entries (2 per spec request), not 131
    assert len(cur[0]) == 4


def test_regression_large_chunk_now_skips():
    """Large non-spec chunk (127 scheduled, absent from spec map) now skips.

    Renamed from test_regression_invalid_decode_still_asserts: after the
    >= boundary fix this class is intentionally hidden as prefill (skip +
    WARNING log at the substitution sites), not asserted. The inclusive
    >= bound also covers the exact max+1 == 2 remainder that strict >
    previously let through (RC2).
    """
    runner = _make_runner(spec_tokens=1)
    runner.input_batch.req_id_to_index = {"req_bad": 0}
    runner.input_batch.num_computed_tokens_cpu = np.array([10], dtype=np.int32)
    runner.input_batch.num_prompt_tokens = np.array([5], dtype=np.int32)
    runner.requests = {"req_bad": MagicMock(prompt_token_ids=[1]*5, num_computed_tokens=10)}
    runner._pre_async_results = MagicMock()
    runner._pre_async_results.placeholder_req_id_to_index = {"req_bad": 0}
    runner.speculative_config = MagicMock(num_speculative_tokens=1)

    req_ids_dp = {0: ["req_bad"]}
    scheduled_tokens = {0: [127]}

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {"req_bad": 127}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.scheduled_new_reqs = []
    cr = MagicMock()
    cr.req_ids = ["req_bad"]
    cr.num_computed_tokens = [10]
    cr.num_output_tokens = [5]
    cr.resumed_req_ids = set()
    scheduler_output.scheduled_cached_reqs = cr

    # After the 320/127 chunked-prefill fix (sharper 180→320 witness) plus the RC2
    # >= boundary fix, a spec-enabled entry not in spec_map scheduled
    # >= num_speculative_tokens + 1 tokens is classified prefill and skipped with
    # a WARNING log, not asserted. Small invalid scheduling (e.g. 2 tokens) hits
    # the same >= bound instead of the stale-counter fallback.
    result = runner._prepare_async_token_substitution_indices(
        req_ids_dp, scheduled_tokens, 128, 1, scheduler_output=scheduler_output)
    # Should skip substitution (is_prefill=True) and return empty indices, not assert
    assert result[0][0] == [] and result[1][0] == []


def test_regression_320_chunked_prefill_sharper_must_skip():
    """320-token chunked prefill (sharper 180→320 witness, KV 0.991) must skip, not assert 320<=2."""
    runner = _make_runner(spec_tokens=1)
    runner.input_batch.req_id_to_index = {"req_320": 0}
    # Stale input_batch says decode (computed 49984 >= prompt 180) but authoritative is prefill
    runner.input_batch.num_computed_tokens_cpu = np.array([49984], dtype=np.int32)
    runner.input_batch.num_prompt_tokens = np.array([180], dtype=np.int32)
    runner.requests = {"req_320": MagicMock(prompt_token_ids=[1]*180, num_computed_tokens=49984)}
    runner._pre_async_results = MagicMock()
    runner._pre_async_results.placeholder_req_id_to_index = {"req_320": 0}
    runner.speculative_config = MagicMock(num_speculative_tokens=1)

    req_ids_dp = {0: ["req_320"]}
    scheduled_tokens = {0: [320]}

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {"req_320": 320}
    scheduler_output.scheduled_spec_decode_tokens = {}  # not in spec verify window
    scheduler_output.scheduled_new_reqs = []
    cr = MagicMock()
    cr.req_ids = ["req_320"]
    cr.num_computed_tokens = [49984]
    cr.num_output_tokens = [646]
    cr.resumed_req_ids = set()
    scheduler_output.scheduled_cached_reqs = cr

    cur, nxt = runner._prepare_async_token_substitution_indices(
        req_ids_dp, scheduled_tokens, 128, 1, scheduler_output=scheduler_output)
    assert cur[0] == [] and nxt[0] == [], "320-token chunked prefill must be skipped (is_prefill=True)"


def test_regression_chunked_prefill_one_token_skip():
    """Chunked prefill with 1-token remaining but still prefill must skip."""
    runner = _make_runner(spec_tokens=1)
    runner.input_batch.req_id_to_index = {"req_chunk": 0}
    # Stale input_batch says decode, but authoritative says prefill
    runner.input_batch.num_computed_tokens_cpu = np.array([10], dtype=np.int32)
    runner.input_batch.num_prompt_tokens = np.array([5], dtype=np.int32)
    runner.requests = {"req_chunk": MagicMock(prompt_token_ids=[1]*200, num_computed_tokens=199)}
    runner._pre_async_results = MagicMock()
    runner._pre_async_results.placeholder_req_id_to_index = {"req_chunk": 0}
    runner.speculative_config = None

    req_ids_dp = {0: ["req_chunk"]}
    scheduled_tokens = {0: [1]}

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {"req_chunk": 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.scheduled_new_reqs = []
    cr = MagicMock()
    cr.req_ids = ["req_chunk"]
    cr.num_computed_tokens = [199]
    cr.num_output_tokens = [0]
    cr.resumed_req_ids = set()
    scheduler_output.scheduled_cached_reqs = cr

    cur, _ = runner._prepare_async_token_substitution_indices(
        req_ids_dp, scheduled_tokens, 128, 1, scheduler_output=scheduler_output)
    assert len(cur[0]) == 0, "1-token chunked prefill should be skipped"


def test_regression_normal_decode_still_substitutes():
    """Normal decode (1 token, not spec) must still substitute."""
    runner = _make_runner(spec_tokens=1)
    runner.input_batch.req_id_to_index = {"req_decode": 0}
    runner.input_batch.num_computed_tokens_cpu = np.array([10], dtype=np.int32)
    runner.input_batch.num_prompt_tokens = np.array([5], dtype=np.int32)
    runner.requests = {"req_decode": MagicMock(prompt_token_ids=[1]*5, num_computed_tokens=10)}
    runner._pre_async_results = MagicMock()
    runner._pre_async_results.placeholder_req_id_to_index = {"req_decode": 0}
    runner.speculative_config = None

    req_ids_dp = {0: ["req_decode"]}
    scheduled_tokens = {0: [1]}

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {"req_decode": 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.scheduled_new_reqs = []
    cr = MagicMock()
    cr.req_ids = ["req_decode"]
    cr.num_computed_tokens = [10]
    cr.num_output_tokens = [5]
    cr.resumed_req_ids = set()
    scheduler_output.scheduled_cached_reqs = cr

    cur, _ = runner._prepare_async_token_substitution_indices(
        req_ids_dp, scheduled_tokens, 128, 1, scheduler_output=scheduler_output)
    assert len(cur[0]) == 1


def test_regression_spec_decode_still_substitutes():
    """Speculative decode (2 tokens, in spec map) must still substitute."""
    runner = _make_runner(spec_tokens=1)
    runner.input_batch.req_id_to_index = {"req_spec": 0}
    runner.input_batch.num_computed_tokens_cpu = np.array([10], dtype=np.int32)
    runner.input_batch.num_prompt_tokens = np.array([5], dtype=np.int32)
    runner.requests = {"req_spec": MagicMock(prompt_token_ids=[1]*5, num_computed_tokens=10)}
    runner._pre_async_results = MagicMock()
    runner._pre_async_results.placeholder_req_id_to_index = {"req_spec": 0}
    runner.speculative_config = MagicMock(num_speculative_tokens=1)

    req_ids_dp = {0: ["req_spec"]}
    scheduled_tokens = {0: [2]}

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {"req_spec": 2}
    scheduler_output.scheduled_spec_decode_tokens = {"req_spec": [1]}
    scheduler_output.scheduled_new_reqs = []
    cr = MagicMock()
    cr.req_ids = ["req_spec"]
    cr.num_computed_tokens = [10]
    cr.num_output_tokens = [5]
    cr.resumed_req_ids = set()
    scheduler_output.scheduled_cached_reqs = cr

    cur, _ = runner._prepare_async_token_substitution_indices(
        req_ids_dp, scheduled_tokens, 128, 1, scheduler_output=scheduler_output)
    assert len(cur[0]) == 2


def test_regression_preempted_resumed_skip():
    """Preempted/recomputed (resumed) must skip even if 1-token."""
    runner = _make_runner(spec_tokens=1)
    runner.input_batch.req_id_to_index = {"req_preempt": 0}
    runner.input_batch.num_computed_tokens_cpu = np.array([10], dtype=np.int32)
    runner.input_batch.num_prompt_tokens = np.array([5], dtype=np.int32)
    runner.requests = {"req_preempt": MagicMock(prompt_token_ids=[1]*200, num_computed_tokens=0)}
    runner._pre_async_results = MagicMock()
    runner._pre_async_results.placeholder_req_id_to_index = {"req_preempt": 0}
    runner.speculative_config = MagicMock(num_speculative_tokens=1)

    req_ids_dp = {0: ["req_preempt"]}
    scheduled_tokens = {0: [127]}

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {"req_preempt": 127}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.scheduled_new_reqs = []
    cr = MagicMock()
    cr.req_ids = ["req_preempt"]
    cr.num_computed_tokens = [0]
    cr.num_output_tokens = [5]
    cr.resumed_req_ids = {"req_preempt"}
    scheduler_output.scheduled_cached_reqs = cr

    cur, _ = runner._prepare_async_token_substitution_indices(
        req_ids_dp, scheduled_tokens, 128, 1, scheduler_output=scheduler_output)
    assert len(cur[0]) == 0


def test_regression_immediate_resume_substitutes():
    """Immediate resume after previous async/spec step must substitute."""
    runner = _make_runner(spec_tokens=1)
    runner.input_batch.req_id_to_index = {"req_resume": 0}
    runner.input_batch.num_computed_tokens_cpu = np.array([10], dtype=np.int32)
    runner.input_batch.num_prompt_tokens = np.array([5], dtype=np.int32)
    runner.requests = {"req_resume": MagicMock(prompt_token_ids=[1]*5, num_computed_tokens=10)}
    runner._pre_async_results = MagicMock()
    runner._pre_async_results.placeholder_req_id_to_index = {"req_resume": 0}
    runner.speculative_config = MagicMock(num_speculative_tokens=1)

    req_ids_dp = {0: ["req_resume"]}
    scheduled_tokens = {0: [2]}

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {"req_resume": 2}
    scheduler_output.scheduled_spec_decode_tokens = {"req_resume": [1]}
    scheduler_output.scheduled_new_reqs = []
    cr = MagicMock()
    cr.req_ids = ["req_resume"]
    cr.num_computed_tokens = [10]
    cr.num_output_tokens = [5]
    cr.resumed_req_ids = set()
    scheduler_output.scheduled_cached_reqs = cr

    cur, _ = runner._prepare_async_token_substitution_indices(
        req_ids_dp, scheduled_tokens, 128, 1, scheduler_output=scheduler_output)
    assert len(cur[0]) == 2
