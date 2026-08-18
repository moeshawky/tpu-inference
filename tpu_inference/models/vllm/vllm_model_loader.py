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

import os
import shutil
import subprocess

import regex as re
import torch
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.runai_streamer_loader import \
    RunaiModelStreamerLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading)
from vllm.utils.torch_utils import set_default_torch_dtype

from tpu_inference.layers.vllm.quantization.base import VllmQuantizationMethod


def _drop_safetensors_page_cache(path: str) -> None:
    """Release consumed checkpoint pages without touching the model tensors."""
    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            posix_fadvise(fd, 0, 0, dontneed)
        finally:
            os.close(fd)
    except OSError:
        # Cache eviction is an optimization; never turn a successful weight
        # load into a failure on filesystems that do not support the hint.
        return


class _CacheDroppingSafeOpen:
    """Delegate safetensors access and drop the file cache on context exit."""

    def __init__(self, inner, path: str):
        self._inner = inner
        self._path = path

    def __enter__(self):
        return self._inner.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self._inner.__exit__(exc_type, exc_value, traceback)
        finally:
            _drop_safetensors_page_cache(self._path)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_safetensors_page_cache_guard() -> None:
    """Bound NFS page-cache growth inside Kaggle's memory cgroup.

    The standard lazy iterator mmap-reads directly from Kaggle's NFS mount.
    On this runtime, DONTNEED is advisory and the NFS pages remain charged to
    the notebook cgroup. Stage one shard at a time through ``dd iflag=direct``
    into tmpfs instead; the staged file is removed before the next shard.
    """
    from vllm.model_executor.model_loader import default_loader, weight_utils

    if getattr(weight_utils, "_tpu_page_cache_guard_installed", False):
        return
    original_safe_open = weight_utils.safe_open
    original_iterator = weight_utils.safetensors_weights_iterator

    def safe_open_with_cache_guard(path, *args, **kwargs):
        return _CacheDroppingSafeOpen(
            original_safe_open(path, *args, **kwargs), os.fspath(path))

    def direct_io_iterator(
        hf_weights_files,
        use_tqdm_on_load,
        safetensors_load_strategy=None,
        local_expert_ids=None,
        *,
        safetensors_prefetch_num_threads=1,
        safetensors_prefetch_block_size=1024 * 1024,
    ):
        stage_dir = "/dev/shm/tpu_inference_safetensors"
        os.makedirs(stage_dir, exist_ok=True)
        for index, source in enumerate(hf_weights_files):
            staged = os.path.join(
                stage_dir, f"shard-{os.getpid()}-{index}.safetensors")
            try:
                try:
                    subprocess.run(
                        [
                            "dd", f"if={os.fspath(source)}", f"of={staged}",
                            "bs=4M", "iflag=direct", "status=none",
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                except (OSError, subprocess.CalledProcessError):
                    # Keep non-Linux/non-NFS environments functional. The
                    # fallback is correct but may retain ordinary file cache.
                    shutil.copyfile(os.fspath(source), staged)
                # safetensors tensors are mmap-backed. Clone each tensor
                # before yielding so vLLM's parameter storage is anonymous
                # memory rather than a retained tmpfs mapping. Without this,
                # unlinking the staged shard leaves its pages charged until
                # the model dies.
                for name, tensor in original_iterator(
                    [staged],
                    use_tqdm_on_load,
                    safetensors_load_strategy,
                    local_expert_ids=local_expert_ids,
                    safetensors_prefetch_num_threads=(
                        safetensors_prefetch_num_threads),
                    safetensors_prefetch_block_size=(
                        safetensors_prefetch_block_size),
                ):
                    detached_tensor = tensor.clone()
                    del tensor
                    yield name, detached_tensor
                    del detached_tensor
            finally:
                try:
                    os.unlink(staged)
                except FileNotFoundError:
                    pass

    weight_utils.safe_open = safe_open_with_cache_guard
    weight_utils.safetensors_weights_iterator = direct_io_iterator
    # default_loader imported the iterator by name, so patch that binding too.
    default_loader.safetensors_weights_iterator = direct_io_iterator
    weight_utils._tpu_page_cache_guard_installed = True


def attach_incremental_weight_loader(model: torch.nn.Module) -> None:
    _install_safetensors_page_cache_guard()
    """
    Traverses the model and overrides the weight_loader of each parameter to support incremental loading.
    This allows processing and sharding of weights after all weights for a module have been loaded.
    """

    def create_weight_loader(layer, original_loader, layer_name, param_name):

        def weight_loader_wrapper(param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, *args,
                                  **kwargs):
            # Loading the weight
            res = original_loader(param, loaded_weight, *args, **kwargs)

            # Processing and sharding
            # Incremental processing and sharding for supported layers.
            # Currently only unquantized and fp8 linear and moe layers supported.
            quant_method = getattr(layer, "quant_method", None)
            if isinstance(quant_method, VllmQuantizationMethod):
                quant_method.maybe_process_weights(layer, param_name, args,
                                                   kwargs)

            return res

        return weight_loader_wrapper

    for name, module in model.named_modules():
        # Weight loader will be invoked multiple times for module. In order to determine when all the weights are loaded,
        # we need to keep track of the loaded weights for each module.
        module._loaded_weights = set()
        for param_name, param in module.named_parameters(recurse=False):
            # Omit parameters that do not have a weight_loader
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue
            setattr(
                param, "weight_loader",
                create_weight_loader(module, original_loader, name,
                                     param_name))


@register_model_loader("tpu_streaming_loader")
class IncrementalModelLoader(DefaultModelLoader):
    """
    Model loader that supports incremental weight loading and sharding.

    This loader is needed to inject the `attach_incremental_weight_loader` logic
    before the actual weight loading begins. This allows us to wrap the
    parameter weight loaders so that weights are sharded to TPU and freed from
    CPU memory as soon as a layer is fully loaded, rather than waiting for the
    entire model to be loaded into CPU memory first.
    """

    def __init__(self, load_config: LoadConfig):
        load_config.load_format = "auto"
        super().__init__(load_config)

    def load_model(self,
                   vllm_config: VllmConfig,
                   model_config: ModelConfig,
                   prefix: str = "") -> torch.nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (device_config.device
                       if load_config.device is None else load_config.device)
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            # Override weight loader logic of each parameter to support incremental loading.
            attach_incremental_weight_loader(model)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()


@register_model_loader("runai_streamer")
class RunaiIncrementalModelLoader(RunaiModelStreamerLoader):
    """Model loader that supports both RunAI streaming and incremental weight sharding."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def _prepare_weights(self, model_name_or_path: str,
                         revision: str | None) -> list[str]:
        hf_weights_files = super()._prepare_weights(model_name_or_path,
                                                    revision)
        hf_weights_files.sort(key=lambda f: [
            int(s) if s.isdigit() else s
            for s in re.split(r"(\d+)", os.path.basename(f))
        ])
        return hf_weights_files

    def load_model(self,
                   vllm_config: VllmConfig,
                   model_config: ModelConfig,
                   prefix: str = "") -> torch.nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (device_config.device
                       if load_config.device is None else load_config.device)
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            # Override weight loader logic of each parameter to support incremental loading.
            attach_incremental_weight_loader(model)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
