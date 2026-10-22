"""Compatibility for torch 2.7.1's built-in TMA autotuning launcher.

KernelArgs.python_argdefs orders buffers, symbolic sizes, then workspace.
TritonBenchmarkRequest.make_run_fn instead puts workspace before symbolic
sizes. A dynamic TMA GEMM therefore passes a tensor to an integer argument.
Keep the upstream allocation/initialization/stream behavior and insert the
workspace at the position recorded in the actual generated kernel signature.
This affects compiler algorithm selection, not the generated training kernel.
"""
import inspect

import torch


def install_tma_workspace_compat():
    if torch.__version__.split('+')[0] != '2.7.1':
        return
    from torch._dynamo.device_interface import get_interface_for_device
    from torch._inductor.autotune_process import TritonGPUBenchmarkRequest
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.codegen.common import WorkspaceZeroMode

    cls = TritonGPUBenchmarkRequest
    if getattr(cls, '_bagel_workspace_order_fixed', False):
        return
    original = cls.make_run_fn

    def make_run_fn(self, *input_tensors, output_tensor):
        if self.workspace_arg is None:
            return original(self, *input_tensors, output_tensor=output_tensor)
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        kernel = getattr(mod, self.kernel_name)
        workspace = self.workspace_arg
        names = kernel.fn.arg_names
        if workspace.inner_name not in names:
            return original(self, *input_tensors, output_tensor=output_tensor)
        position = names.index(workspace.inner_name)
        run_method = kernel.run
        run_method.__self__.with_bandwidth_info = False
        warmup = {'warmup': False} if 'warmup' in inspect.signature(run_method).parameters else {}
        interface = get_interface_for_device(output_tensor.device.type)
        stream = interface.get_raw_stream(self.output_tensor_meta.device.index)

        def run_with_workspace():
            tensor = torch.empty_strided((workspace.count,), (1,), dtype=torch.uint8,
                                         device=output_tensor.device)
            if workspace.zero_mode != WorkspaceZeroMode.UNINITIALIZED:
                tensor.zero_()
            args = [*input_tensors, output_tensor, *self.extra_args]
            args.insert(position, tensor)
            run_method(*args, **warmup, stream=stream, benchmark_run=True)

        return run_with_workspace

    cls.make_run_fn = make_run_fn
    cls._bagel_workspace_order_fixed = True
