"""Length arithmetic and legal fixed AOT exclusions during convergence."""

import unittest
from orinfer_build.tuning.row_convergence import divide_rows
from orinfer_build.contracts.row_selection import split_fixed


class ConvergenceTests(unittest.TestCase):
    def test_metadata_certificates_keep_private_offsets_and_radix_order(self):
        from orinfer_build.compiler.host_compiler import private_roles

        roles, offsets = private_roles(
            "a = Pointers[(((int64_t)lane) * (int64_t)624)];\n"
            "b = Pointers[((((int64_t)lane) * (int64_t)624) + (int64_t)3)];\n"
            "void* Query_row = orin_row_pointer((a + (uint64_t)108544));\n"
            "void* Position_row = orin_row_pointer(b);",
            ["Scratch", "Length", "MRope", "Position"],
        )
        self.assertEqual(
            roles, dict(Query=("private", "Scratch"), Position=("private", "Position"))
        )
        self.assertEqual(offsets, dict(Query=108544, Position=0))

    def test_dynamic_tuning_covers_the_allocation_instead_of_one_short_sample(self):
        from orinfer_build.tuning.projection_tuning import profile_rows

        for capacity in (1, 7, 128, 1024, 4096):
            samples = profile_rows(capacity)
            self.assertEqual((samples[0], samples[-1]), (1, capacity))
            self.assertEqual(samples, sorted(set(samples)))
            self.assertTrue(all(1 <= s <= capacity for s in samples))
            if capacity >= 128:
                self.assertTrue(any(17 < s < capacity for s in samples))

    def test_adjacent_merge_keeps_gaps_arguments_and_launch_contracts(self):
        import copy
        from orinfer_build.tuning.row_convergence import coalesce_adjacent_selections

        def entry(first, last, scalar=99):
            name = f"body/k0/rows/{first}-{last}"
            return dict(name=name, module="same", args=[dict(kind="i32", value=scalar)])

        kernels = [entry(1, 1), entry(2, 3), entry(5, 7), entry(8, 8, 100), entry(9, 10, 100)]
        launches = [
            dict(name=k["name"], capacity=128, arguments=[dict(index=0, value=dict(op="rows"))])
            for k in kernels
        ]
        package = dict(
            kernels=kernels,
            dynamic_batch_kernels=launches,
            plan_kernel_contracts=[k["name"] for k in kernels],
        )
        self.assertEqual(coalesce_adjacent_selections(package), 2)
        self.assertEqual(
            sorted(k["name"] for k in package["kernels"]),
            ["body/k0/rows/1-3", "body/k0/rows/5-7", "body/k0/rows/8-10"],
        )
        before = copy.deepcopy(package)
        self.assertEqual(coalesce_adjacent_selections(package), 0)
        self.assertEqual(package, before)
        package["kernels"].append(entry(2, 4))
        package["dynamic_batch_kernels"].append(dict(launches[0], name="body/k0/rows/2-4"))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            coalesce_adjacent_selections(package)

    def test_immutable_scalars_remain_constant_in_dynamic_launches(self):
        from orinfer_build.contracts.row_contract import expression, contract

        self.assertEqual(
            expression("base_exp", constants=dict(base_exp=99)), dict(op="constant", value=99)
        )
        host = dict(
            launch_expressions={
                **{f"gridDim{a}": "1" for a in "XYZ"},
                **{f"blockDim{a}": "1" for a in "XYZ"},
                "sharedMemBytes": "0",
            },
            ordered_arguments=[],
        )
        kernel = dict(name="test", grid=[1, 1, 1], block=[1, 1, 1], shared_memory_bytes=0, args=[])
        for constants in (dict(m=1), dict(shift=True), dict(shift=-1), dict(shift=2**31)):
            with self.assertRaisesRegex(ValueError, "immutable"):
                contract(kernel, host, constants=constants)

    def test_merger_scales_grid_and_scalar_in_one_row_contract(self):
        def evaluate(expr, rows):
            if expr["op"] == "rows":
                return rows
            if expr["op"] == "constant":
                return expr["value"]
            a, b = evaluate(expr["lhs"], rows), evaluate(expr["rhs"], rows)
            return a + b if expr["op"] == "add" else a // b

        scalar = divide_rows(dict(op="rows"), 4)
        grid = divide_rows(
            dict(
                op="divide",
                lhs=dict(op="add", lhs=dict(op="rows"), rhs=dict(op="constant", value=63)),
                rhs=dict(op="constant", value=64),
            ),
            4,
        )
        for patches in [4, 140, 256, 540, 8192, 32768]:
            self.assertEqual(evaluate(scalar, patches), patches // 4)
            self.assertEqual(evaluate(grid, patches), (patches // 4 + 63) // 64)
        for patches in [1, 2, 3]:
            self.assertGreater(evaluate(grid, patches), 0)

    def test_exact_precompilations_do_not_steal_neighbouring_rows(self):
        self.assertEqual(split_fixed(1, 40, {"1": "a", "2": "b", "32": "c"}), [(3, 31), (33, 40)])
