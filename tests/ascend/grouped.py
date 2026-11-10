"""Offline CANN 9.2 grouped-matmul probes using the runtime's physical views."""

import ctypes as C
import math
import os
from pathlib import Path


class GroupedMatmul:
    """Retain repeatable ACLNN metadata until all probe Graphs are released.

    Tensor-list metadata owns its child tensors, as in the Rust runtime. Call
    close only after the last device completion and destruction of its Graphs.
    """

    def __init__(
        self,
        activation,
        weight,
        scale,
        assist,
        token_scale,
        groups,
        output,
        *,
        rows,
        reduction,
        columns,
        experts,
        split=False,
        expected_rows=16,
    ):
        import torch

        self.storage = [activation, weight, scale, assist, token_scale, groups, output]
        self.tensors, self.lists = [], []
        self.executor = C.c_void_p()
        root = Path(os.environ["ASCEND_HOME_PATH"]) / "lib64"
        self.meta, self.ops = (
            C.CDLL(str(root / "libnnopbase.so")),
            C.CDLL(str(root / "libopapi.so")),
        )
        P, I, U = C.c_void_p, C.c_int64, C.c_uint64
        declarations = {
            "aclCreateTensor": (
                [C.POINTER(I), U, C.c_int32, C.POINTER(I), I, C.c_int32, C.POINTER(I), U, P],
                P,
            ),
            "aclCreateTensorList": ([C.POINTER(P), U], P),
            "aclCreateIntArray": ([C.POINTER(I), U], P),
            "aclSetAclOpExecutorRepeatable": ([P], C.c_int32),
            "aclDestroyAclOpExecutor": ([P], C.c_int32),
            "aclDestroyTensorList": ([P], C.c_int32),
            "aclDestroyTensor": ([P], C.c_int32),
            "aclDestroyIntArray": ([P], C.c_int32),
        }
        for name, (args, result) in declarations.items():
            fn = getattr(self.meta, name)
            fn.argtypes, fn.restype = args, result

        def tensor(data, shape, dtype, bits, storage=None, transpose=False):
            if math.prod(shape) * bits > data.numel() * data.element_size() * 8:
                raise ValueError("Grouped tensor exceeds retained device storage")
            strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
            if transpose:
                strides[1:] = [1, shape[1]]
            physical = storage or shape
            handle = self.meta.aclCreateTensor(
                (I * len(shape))(*shape),
                len(shape),
                dtype,
                (I * len(shape))(*strides),
                0,
                29 if storage else 2,
                (I * len(physical))(*physical),
                len(physical),
                data.data_ptr(),
            )
            if not handle:
                raise RuntimeError("aclCreateTensor returned null")
            self.tensors.append(handle)
            return handle

        def tensor_list(handle):
            result = self.meta.aclCreateTensorList((P * 1)(handle), 1)
            if not result:
                raise RuntimeError("aclCreateTensorList returned null")
            self.tensors.remove(handle)
            self.lists.append(result)
            return result

        views = [
            tensor(activation, [rows, reduction], 29 if split else 2, 4 if split else 8),
            tensor(
                weight,
                [experts, reduction, columns],
                29,
                4,
                [experts, reduction // 64, columns // 16, 16, 64],
                transpose=split,
            ),
            tensor(scale, [experts, 1, columns], 10, 64),
            tensor(assist, [experts, columns], 0, 32),
            tensor(token_scale, [rows], 0, 32),
            tensor(output, [rows, columns], 1 if split else 27, 16),
        ]
        lists = [tensor_list(v) for v in views]
        group = tensor(groups, [experts], 9, 64)
        self.tuning_values = (I * 2)(expected_rows, 1)
        self.tuning = self.meta.aclCreateIntArray(self.tuning_values, 2)
        if not self.tuning:
            raise RuntimeError("aclCreateIntArray returned null")
        prepare = self.ops.aclnnGroupedMatmulV5GetWorkspaceSize
        prepare.argtypes = [P] * 12 + [I] * 4 + [P] * 4 + [C.POINTER(U), C.POINTER(P)]
        prepare.restype = C.c_int32
        size = U()
        status = prepare(
            lists[0],
            lists[1],
            None if split else lists[3],
            lists[2],
            None,
            None,
            None,
            lists[4],
            group,
            None,
            None,
            None,
            2,
            0,
            1,
            0,
            self.tuning,
            lists[5],
            None,
            None,
            C.byref(size),
            C.byref(self.executor),
        )
        if status or not self.executor:
            raise RuntimeError(f"Grouped preparation failed: {status}")
        status = self.meta.aclSetAclOpExecutorRepeatable(self.executor)
        if status:
            raise RuntimeError(f"Grouped repeatable executor failed: {status}")
        self.workspace = torch.empty(size.value, dtype=torch.uint8, device=activation.device)
        self.workspace_bytes = size.value
        self.execute = self.ops.aclnnGroupedMatmulV5
        self.execute.argtypes, self.execute.restype = [P, U, P, P], C.c_int32

    def __call__(self):
        import torch

        status = self.execute(
            self.workspace.data_ptr(),
            self.workspace_bytes,
            self.executor,
            torch.npu.current_stream()._as_parameter_,
        )
        if status:
            raise RuntimeError(f"Grouped execution failed: {status}")

    def close(self):
        if self.executor:
            self.meta.aclDestroyAclOpExecutor(self.executor)
            self.executor = C.c_void_p()
        for handle in self.lists:
            self.meta.aclDestroyTensorList(handle)
        self.lists.clear()
        for handle in self.tensors:
            self.meta.aclDestroyTensor(handle)
        self.tensors.clear()
        if self.tuning:
            self.meta.aclDestroyIntArray(self.tuning)
            self.tuning = None
        self.storage.clear()
        self.workspace = None
