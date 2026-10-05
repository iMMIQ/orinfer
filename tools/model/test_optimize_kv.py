"""Offline contract tests for direct, demand-mapped KV publication."""
import unittest
from tools.model.optimize_kv import transform


class KvContractTests(unittest.TestCase):
    def fixture(self):
        buffers=[dict(name=n,dtype='f16',shape=[4,128,4,256],data=None) for n in ('L3_KPages','L3_VPages','MtpKPages','MtpVPages')]
        buffers += [dict(name=n) for n in ('Kcontig','Vcontig','Pages')]
        wrapper=dict(architecture='qwen3_5',buffer_scopes={b['name']:'sequence' for b in buffers},
            metadata=dict(buffers=buffers,max_context=512,reset_buffers=['L3_KPages','L3_VPages']))
        package=dict(kernels=[dict(name='prefill_m512/layer3/k4',args=[dict(kind='buffer',name=n) for n in ('Kcontig','Pages','L3_KPages','L3_VPages','Vcontig')]),
            dict(name='prefill_m512/layer3/k5',args=[dict(kind='buffer',name=n) for n in ('Kcontig','Vcontig','FullQ')]),
            dict(name='decode/layer3/k4',args=[dict(kind='buffer',name='L3_KPages')])])
        return wrapper,package

    def test_direct_reads_keep_slot_identity_and_remove_scratch(self):
        w,p=transform(*self.fixture())
        self.assertEqual([k['name'] for k in p['kernels']],['prefill_m512/layer3/k5','decode/layer3/k4'])
        self.assertEqual([a['name'] for a in p['kernels'][0]['args']],['L3_KPages','L3_VPages','FullQ'])
        self.assertNotIn('Kcontig',w['buffer_scopes'])
        self.assertEqual(w['metadata']['kv_cache']['buffers'],{n:2048 for n in ('L3_KPages','L3_VPages','MtpKPages','MtpVPages')})

    def test_rejects_gather_in_an_unknown_recipe(self):
        w,p=self.fixture();p['kernels'][0]['name']='decode/layer3/k4'
        with self.assertRaisesRegex(ValueError,'gather recipe'): transform(w,p)

    def test_rejects_payload_geometry_or_dtype_before_publication(self):
        w,p=self.fixture();w['metadata']['buffers'][0]['dtype']='i8'
        with self.assertRaisesRegex(ValueError,'Invalid KV payload'): transform(w,p)
        w,p=self.fixture();w['metadata']['buffers'][0]['shape'][0]=3
        with self.assertRaisesRegex(ValueError,'Invalid KV payload'): transform(w,p)


if __name__=='__main__': unittest.main()
