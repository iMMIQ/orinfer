import unittest
from tools.model.flash_next.speculation import greedy_commit,clip_outputs,verification_size
from tools.model.flash_next.roles import role
from tools.quantization.flash_next_aux import tasks,policy
from tools.model.flash_next.weights import coverage
from tools.model.flash_next.policy import vocabulary,AdaptiveDepth


class SpeculationTests(unittest.TestCase):
    def test_draft_vocabulary_preserves_ids_ties_and_required_tokens(self):
        selected=vocabulary({400:20,300:20,600:1},{999},256,1024)
        self.assertEqual(selected,sorted(set(selected)))
        self.assertEqual(len(selected),256)
        self.assertTrue({300,400,600,999}.issubset(selected))
        self.assertEqual(selected,vocabulary({600:1,300:20,400:20},{999},256,1024))
        for args in (({},set(),257,1024),({},set(),128,1024),({1024:1},set(),256,1024),
                     ({},set(range(257)),256,1024)):
            with self.assertRaises(ValueError):vocabulary(*args)

    def test_adaptive_uses_throughput_not_acceptance_alone(self):
        policy=AdaptiveDepth(interval=1,alpha=1.)
        # Long tiers accept everything but are much more expensive per token.
        policy.observe(7,7,7,1000)
        policy.observe(3,3,3,100)
        policy.observe(1,1,1,1)
        self.assertEqual(policy.choose(),1)
        self.assertEqual(policy.current,1)
        with self.assertRaises(ValueError):policy.observe(1,1,2,1)

    def test_adaptive_probes_reset_and_clipped_tails(self):
        policy=AdaptiveDepth(interval=2)
        policy.observe(3,1,1,10)
        self.assertEqual(policy.rounds,0)
        for _ in range(2):policy.observe(3,3,2,100)
        self.assertEqual(policy.choose(),1)
        self.assertEqual(policy.choose(),3)
        self.assertEqual(AdaptiveDepth().rounds,0)

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
