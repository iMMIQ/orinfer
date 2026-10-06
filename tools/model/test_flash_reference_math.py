"""CPU checks including an independent Transformers delta-rule oracle."""
import unittest

import torch

from tools.model.flash_reference_math import attention, causal_conv, delta_rule, neox, norm, read_streams, write_streams


class ReferenceMathTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(20261002);torch.set_num_threads(2)

    def test_delta_against_transformers_chunk_solver_and_nonzero_restore(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        q,k = (torch.randn(7,2,4) for _ in range(2));v = torch.randn(7,6,3)
        decay = -torch.rand(7,6);beta = torch.rand(7,6);initial = torch.randn(6,4,3)
        expected,final = torch_chunk_gated_delta_rule(q.repeat_interleave(3,1)[None],k.repeat_interleave(3,1)[None],
            v[None],decay[None],beta[None],chunk_size=4,initial_state=initial[None],
            output_final_state=True,use_qk_l2norm_in_kernel=True)
        actual,state = delta_rule(q,k,v,decay,beta,state=initial)
        torch.testing.assert_close(actual,expected[0],rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(state,final[0],rtol=2e-5,atol=2e-6)
        first,prefix = delta_rule(q[:3],k[:3],v[:3],decay[:3],beta[:3],state=initial)
        second,restored = delta_rule(q[3:],k[3:],v[3:],decay[3:],beta[3:],state=prefix)
        torch.testing.assert_close(torch.cat((first,second)),actual,rtol=0,atol=0)
        torch.testing.assert_close(restored,state,rtol=0,atol=0)

    def test_causal_request_isolation_and_half_pair_rotation(self):
        x = torch.tensor([[[1.,2.,3.,4.,5.,6.]],[[1.,2.,3.,4.,5.,6.]]])
        rotated = neox(x,4)
        self.assertEqual(rotated[1,0,4:].tolist(),[5.,6.])
        self.assertAlmostEqual(float(rotated[1,0,0]),float(torch.cos(torch.tensor(1.))-3*torch.sin(torch.tensor(1.))),places=6)
        self.assertAlmostEqual(float(rotated[1,0,2]),float(3*torch.cos(torch.tensor(1.))+torch.sin(torch.tensor(1.))),places=6)
        torch.testing.assert_close(norm(x,torch.zeros(6)),x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-6))
        qg = torch.randn(5,4,2,8);k,v = (torch.randn(5,2,8) for _ in range(2))
        qw,kw = torch.randn(8),torch.randn(8)
        full = attention(qg,k,v,qw,kw,rotary=4)
        k[3:] = 100;v[3:] = -100;qg[3:] = 200
        changed = attention(qg,k,v,qw,kw,rotary=4)
        torch.testing.assert_close(changed[:3],full[:3],rtol=0,atol=0)

    def test_dilated_impulse_and_zero_rank_hyperconnection(self):
        x = torch.zeros(11,2);x[0,0] = 1
        w = torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.]])
        actual = causal_conv(x,w,dilation=3)
        preactivation = torch.zeros_like(x);preactivation[[0,3,6,9],0] = torch.tensor([4.,3.,2.,1.])
        torch.testing.assert_close(actual,torch.nn.functional.silu(preactivation),rtol=0,atol=0)
        residual = torch.randn(3,4,8)
        weights = {'norm':torch.zeros(4,8),'down':torch.zeros(2,32),'up':torch.zeros(32,2),'inject':torch.zeros(4,32)}
        mixed,gate = read_streams(residual,weights)
        torch.testing.assert_close(mixed,norm(residual,weights['norm']).mean(1)/2,rtol=0,atol=0)
        torch.testing.assert_close(gate,torch.ones_like(gate),rtol=0,atol=0)
        torch.testing.assert_close(write_streams(residual,mixed,gate),residual+mixed[:,None],rtol=0,atol=0)


if __name__ == '__main__':unittest.main()
