import unittest
from tools.model.upgrade_dynamic_batch import expression, contract


class DynamicContractTests(unittest.TestCase):
    def test_ceildiv_and_unsupported_abi(self):
        result = expression('(M + 31) // 32')
        self.assertEqual(result['op'], 'divide')
        self.assertEqual(result['lhs']['lhs'], {'op':'rows'})
        for source in ['M.real', 'eval(M)', '-1', 'other', 'M ** 2']:
            with self.assertRaises(ValueError): expression(source)

    def test_argument_order_and_fixed_gdn_grid(self):
        host = dict(ordered_arguments=[dict(ctype='ctypes.c_int32',value='M')],
                    launch_expressions=dict(gridDimX='(M + 31) // 32',gridDimY='1',gridDimZ='1',
                                            blockDimX='128',blockDimY='1',blockDimZ='1',sharedMemBytes='0'))
        k = dict(name='batch_m128/layer0/k1',grid=[4,1,1],block=[128,1,1],
                 shared_memory_bytes=0,args=[dict(kind='i32',value=128)])
        result = contract(k,host)
        self.assertEqual(result['arguments'],[dict(index=0,value=dict(op='rows'))])
        k['args'][0]['value'] = 64
        with self.assertRaises(ValueError): contract(k,host)
        host['ordered_arguments'] = [];host['launch_expressions']['gridDimX']='80'
        host['launch_expressions']['gridDimY']='128'
        k.update(name='batch_gdn_m128/layer0/k0',grid=[80,128,1],args=[])
        self.assertEqual(contract(k,host)['grid'][1],dict(op='rows'))


if __name__ == '__main__': unittest.main()
