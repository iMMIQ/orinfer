"""Validate shared prefill scratch geometry and its allocation scope."""
import unittest
from tools.model.stage_kv_prefill import add_workspace

class PrefillWorkspaceTests(unittest.TestCase):
    def fixture(self):
        return dict(architecture='qwen3_5',buffer_scopes={},metadata=dict(max_context=262144,
            buffers=[dict(name=n,dtype='i8',shape=[2048,128,4,256]) for n in ['L3_KPages','L3_VPages']],
            reset_buffers=['L3_KPages','L3_VPages'],kv_cache=dict(direct_prefill=True,demand_mapping=True)))
    def test_one_layer_capacity_and_workspace_scope(self):
        w=add_workspace(self.fixture());m=w['metadata']
        self.assertEqual(m['kv_cache']['prefill_workspace'],dict(PrefillK=2048,PrefillV=2048))
        self.assertEqual(sum(m['kv_cache']['prefill_workspace'].values())*m['max_context'],1<<30)
        self.assertEqual(w['buffer_scopes'],dict(PrefillK='workspace',PrefillV='workspace'))
        self.assertNotIn('PrefillK',m['reset_buffers'])
    def test_rejects_wrong_storage_or_existing_workspace(self):
        w=self.fixture();w['metadata']['buffers'][0]['dtype']='f16'
        with self.assertRaises(ValueError):add_workspace(w)
        w=add_workspace(self.fixture())
        with self.assertRaises(ValueError):add_workspace(w)

if __name__=='__main__':unittest.main()
