"""Address-table packet I/O; request membership is data, not captured pointers."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.cuda.model.independent_rows import POINTER_SOURCE
from kernels.cuda.model.packet_io import output_fragments

IO_COLUMNS = 10


@orin_jit
def inputs(private_stride, context, features, vision):
    base = 128 * private_stride

    @T.prim_func
    def kernel(
        Pointers: T.Tensor((base + 128 * IO_COLUMNS,), T.uint64),
        Embedding: T.Tensor((128, 2560), T.float16),
        Ple: T.Tensor((128, 2560), T.float16),
        m: T.int32,
    ):
        with T.Kernel(m, threads=128) as row:
            T.import_source(POINTER_SOURCE)
            offset = base + row * IO_COLUMNS
            embedding = T.bind(
                T.call_extern("handle", "orin_row_pointer", Pointers[offset]), var=T.ptr("float16")
            )
            embedding_buffer = T.decl_buffer((2560,), "float16", data=embedding)
            ple = T.bind(
                T.call_extern("handle", "orin_row_pointer", Pointers[offset + 1]),
                var=T.ptr("float16"),
            )
            ple_buffer = T.decl_buffer((2560,), "float16", data=ple)
            if vision:
                step = T.bind(
                    T.call_extern("handle", "orin_row_pointer", Pointers[offset + 2]),
                    var=T.ptr("int32"),
                )
                step_buffer = T.decl_buffer((1,), "int32", data=step)
                index = T.bind(
                    T.call_extern("handle", "orin_row_pointer", Pointers[offset + 7]),
                    var=T.ptr("int32"),
                )
                index_buffer = T.decl_buffer((context,), "int32", data=index)
                visual = T.bind(
                    T.call_extern("handle", "orin_row_pointer", Pointers[offset + 6]),
                    var=T.ptr("float16"),
                )
                visual_buffer = T.decl_buffer((features * 2560,), "float16", data=visual)
                token = step_buffer[0]
                feature = T.alloc_local((1,), T.int32)
                feature[0] = -1
                if token >= 0 and token < context:
                    feature[0] = index_buffer[token]
                for col in T.Parallel(2560):
                    Embedding[row, col] = embedding_buffer[col]
                    if feature[0] >= 0 and feature[0] < features:
                        Embedding[row, col] = visual_buffer[feature[0] * 2560 + col]
                    Ple[row, col] = ple_buffer[col]
            else:
                for col in T.Parallel(2560):
                    Embedding[row, col] = embedding_buffer[col]
                    Ple[row, col] = ple_buffer[col]

    return kernel


@orin_jit
def end(private_stride, ring, capture):
    base = 128 * private_stride

    @T.prim_func
    def kernel(
        Pointers: T.Tensor((base + 128 * IO_COLUMNS,), T.uint64),
        Hidden: T.Tensor((128, 10240), T.float16),
        m: T.int32,
    ):
        with T.Kernel(m, threads=128) as row:
            T.import_source(POINTER_SOURCE)
            offset = base + row * IO_COLUMNS
            step = T.bind(
                T.call_extern("handle", "orin_row_pointer", Pointers[offset + 2]),
                var=T.ptr("int32"),
            )
            step_buffer = T.decl_buffer((1,), "int32", data=step)
            position = T.alloc_shared((1,), T.int32)
            if T.get_thread_binding() == 0:
                position[0] = step_buffer[0]
            T.sync_threads()
            if capture:
                target = T.bind(
                    T.call_extern("handle", "orin_row_pointer", Pointers[offset + 8]),
                    var=T.ptr("float16"),
                )
                target_buffer = T.decl_buffer((ring * 10240,), "float16", data=target)
                for col in T.Parallel(10240):
                    target_buffer[(position[0] % ring) * 10240 + col] = Hidden[row, col]
            if T.get_thread_binding() == 0:
                step_buffer[0] = position[0] + 1

    return kernel


@orin_jit
def outputs(private_stride, vocab):
    base = 128 * private_stride
    write_logits, write_selected = output_fragments(vocab, batched=True)

    @T.prim_func
    def kernel(
        Pointers: T.Tensor((base + 128 * IO_COLUMNS,), T.uint64),
        Logits: T.Tensor((128, vocab), T.float32),
        Selected: T.Tensor((128, 2), T.int32),
        m: T.int32,
    ):
        with T.Kernel(m, T.ceildiv(vocab, 512), threads=128) as (row, block):
            T.import_source(POINTER_SOURCE)
            offset = base + row * IO_COLUMNS
            logits = T.bind(
                T.call_extern("handle", "orin_row_pointer", Pointers[offset + 5]),
                var=T.ptr("float32"),
            )
            logits_buffer = T.decl_buffer((vocab,), "float32", data=logits)
            write_logits(Logits, logits_buffer, row, block)
            if block == 0 and T.get_thread_binding() == 0:
                token = T.bind(
                    T.call_extern("handle", "orin_row_pointer", Pointers[offset + 3]),
                    var=T.ptr("int32"),
                )
                token_buffer = T.decl_buffer((1,), "int32", data=token)
                status = T.bind(
                    T.call_extern("handle", "orin_row_pointer", Pointers[offset + 4]),
                    var=T.ptr("int32"),
                )
                status_buffer = T.decl_buffer((1,), "int32", data=status)
                write_selected(Selected, token_buffer, status_buffer, row)

    return kernel
