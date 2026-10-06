import unittest
from tools.model.flash_speculation import greedy_commit,clip_outputs,verification_size
from tools.model.flash_roles import role
from tools.quantization.flash_next_aux import tasks,policy
from tools.model.flash_weights import coverage


class SpeculationTests(unittest.TestCase):
    def test_greedy_accept_reject_and_bonus(self):
        self.assertEqual(greedy_commit([7,8],[7,8,9]),[7,8,9])
        self.assertEqual(greedy_commit([7,8],[2,8,9]),[2])
        self.assertEqual(greedy_commit([7,8],[7,2,9]),[7,2])
        self.assertEqual(greedy_commit([],[2]),[2])
        for target in ([],[1],[1,2,3]):
            with self.assertRaises(ValueError):greedy_commit([1],target)

    def test_budget_eos_and_context_tail(self):
        self.assertEqual(clip_outputs([1,2,3],{2},3),([1,2],'stop'))
        self.assertEqual(clip_outputs([1,2,3],{9},2),([1,2],'length'))
        self.assertEqual(clip_outputs([1,2],{9},3),([1,2],None))
        self.assertEqual(verification_size(2,8,8),3)
        self.assertEqual(verification_size(2,1,8),1)
        self.assertEqual(verification_size(2,8,1),1)
        for args in ((0,8,8),(8,8,8),(1,0,8),(1,8,0)):
            with self.assertRaises(ValueError):verification_size(*args)

    def test_bounded_tail_profiles(self):
        self.assertEqual({verification_size(7,n,32) for n in range(1,9)},{1,2,4,8})
        self.assertEqual({verification_size(5,n,32) for n in range(1,7)},{1,2,4,6})
        self.assertEqual(verification_size(7,8,7),4)
        for depth in range(1,8):
            for remaining in range(1,17):
                for capacity in range(1,17):
                    width=verification_size(depth,remaining,capacity)
                    self.assertLessEqual(width,min(depth+1,remaining,capacity))
                    if min(remaining,capacity)>=depth+1:self.assertEqual(width,depth+1)
                    else:self.assertEqual(width & (width-1),0)

    def test_mtp_roles_and_separate_conversion_coverage(self):
        self.assertEqual(role('mtp.layers.0.self_attn.q_proj.weight'),'blk.48.attn_q.weight')
        self.assertEqual(role('mtp.pre_fc_norm_hidden.weight'),'pre_fc_norm_hidden.weight')
        with self.assertRaises(ValueError):role('mtp.layers.1.self_attn.q_proj.weight')
        class Source:
            weight_map={'mtp.fc_hidden.weight':'x','model.language_model.norm.weight':'x','lm_head.weight':'x'}
            def tensor(self,name):return 'x',0,{'shape':[128,128],'dtype':'BF16'}
        source=Source()
        selected=tasks(source,component='mtp')
        self.assertEqual([t['tensor'] for t in selected],['mtp.fc_hidden.weight'])
        self.assertEqual(policy('mtp.fc_hidden.weight',{'shape':[128,128]}),'original')
        records=[{'tensor':'mtp.fc_hidden.weight','first':0,'count':128,'kind':'original',
                  'group':'aux','source_shape':[128,128],'dtype':'BF16'}]
        self.assertTrue(coverage(source,records,component='mtp')['complete'])
        with self.assertRaises(ValueError):coverage(source,records)


if __name__=='__main__':unittest.main()
