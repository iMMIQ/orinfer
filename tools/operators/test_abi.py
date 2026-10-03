import unittest

from abi import evaluate, parse_host


HOST = '''
def call(kernels, A, C, rows, stream=0):
    config = CUlaunchConfig()
    config.gridDimX = (rows + 15) // 16
    config.blockDimX = 128
    config.sharedMemBytes = 24576
    config.hStream = stream
    arg_values = C.data_ptr(), A.data_ptr(), rows
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32
    res = cuLaunchKernelEx(config, kernels["main_kernel"], (arg_values, arg_types), 0)[0]
'''


class ActualAbiTests(unittest.TestCase):
    def test_actual_generated_order_overrides_primfunc_order(self):
        launch = parse_host(HOST)[0]
        self.assertEqual(launch["symbol"], "main_kernel")
        self.assertEqual([item["value"] for item in launch["ordered_arguments"]],
                         ["C.data_ptr()", "A.data_ptr()", "rows"])
        self.assertEqual(launch["ordered_arguments"][-1]["ctype"], "ctypes.c_int32")
        self.assertEqual(evaluate(launch["launch_expressions"]["gridDimX"], {"rows": 17}), 2)

    def test_rejects_missing_and_mismatched_argument_types(self):
        with self.assertRaises(ValueError):
            parse_host(HOST.replace("ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32",
                                    "ctypes.c_void_p"))
        with self.assertRaises(ValueError):
            parse_host(HOST.replace("    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32\n", ""))

    def test_dimension_binding_cannot_execute_code(self):
        for expression in ("__import__('os')", "rows.__class__", "rows[0]", "True"):
            with self.assertRaises(ValueError):
                evaluate(expression, {"rows": 17})
        with self.assertRaises(KeyError):
            evaluate("undefined + 1", {})


if __name__ == "__main__":
    unittest.main()
