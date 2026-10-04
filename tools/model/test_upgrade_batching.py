import unittest
from tools.model.upgrade_batching import bind


class BatchAbiTests(unittest.TestCase):
    def test_dynamic_shape_is_bound_by_host_abi_order(self):
        export = dict(module={'file': 'kernel.cubin'}, source={'file': 'kernel.cu'},
            host_abi={'file': 'host.txt'}, symbol='kernel_kernel',
            ordered_arguments=[dict(ctype='ctypes.c_int32', value='M'),
                               dict(ctype='ctypes.c_void_p', value='State.data_ptr()'),
                               dict(ctype='ctypes.c_int32', value='batch')],
            launch_expressions=dict(gridDimX='(M + 15) // 16', gridDimY='batch', gridDimZ='1',
                blockDimX='128', blockDimY='1', blockDimZ='1', sharedMemBytes='0'))
        actual = bind(export, 'batch_m4/layer0/k0', {'State': 'L0_State'}, dict(M=4, batch=1))
        self.assertEqual(actual['args'], [dict(kind='i32', value=4),
            dict(kind='buffer', name='L0_State'), dict(kind='i32', value=1)])
        self.assertEqual(actual['grid'], [1, 1, 1])
        self.assertFalse(actual['cooperative'])
        export['ordered_arguments'][0]['ctype'] = 'ctypes.c_double'
        with self.assertRaises(ValueError):
            bind(export, 'invalid', {'State': 'L0_State'}, dict(M=4, batch=1))


if __name__ == '__main__':
    unittest.main()
