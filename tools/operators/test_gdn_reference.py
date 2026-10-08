"""CPU mathematical identity tests; run with Torch in the offline image."""

import unittest

import torch

from gdn_reference import chunked, recurrent


class DeltaRuleIdentityTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(20261002)
        q = torch.randn(2, 2, 3, 4, 5)
        k = torch.randn_like(q)
        q = torch.nn.functional.normalize(q, dim=-1) / 5**0.5
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.randn(2, 6, 3, 4, 7)
        g = -torch.rand(2, 6, 3, 4) * 0.2
        beta = torch.rand_like(g)
        state = torch.randn(2, 6, 5, 7) * 0.2
        return q, k, v, g, beta, state

    def test_chunk_and_token_recurrence_match(self):
        q, k, v, g, beta, state = self.inputs()
        actual, final, _ = chunked(q, k, v, g, beta, state)
        expected, reference = recurrent(
            q.flatten(2, 3),
            k.flatten(2, 3),
            v.flatten(2, 3),
            g.flatten(2, 3),
            beta.flatten(2, 3),
            state,
        )
        torch.testing.assert_close(actual.flatten(2, 3), expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(final, reference, atol=2e-5, rtol=2e-5)

    def test_chunk_resume_and_request_isolation(self):
        q, k, v, g, beta, state = self.inputs()
        expected, final, _ = chunked(q, k, v, g, beta, state)
        first, saved, _ = chunked(
            q[:, :, :1], k[:, :, :1], v[:, :, :1], g[:, :, :1], beta[:, :, :1], state
        )
        rest, resumed, _ = chunked(
            q[:, :, 1:], k[:, :, 1:], v[:, :, 1:], g[:, :, 1:], beta[:, :, 1:], saved
        )
        torch.testing.assert_close(torch.cat((first, rest), dim=2), expected)
        torch.testing.assert_close(resumed, final)
        one, one_final, _ = chunked(q[:1], k[:1], v[:1], g[:1], beta[:1], state[:1])
        torch.testing.assert_close(one, expected[:1])
        torch.testing.assert_close(one_final, final[:1])

    def test_padding_does_not_change_state(self):
        q, k, v, g, beta, state = self.inputs()
        q[:, :, -1, -2:] = 0
        k[:, :, -1, -2:] = 0
        v[:, :, -1, -2:] = 0
        g[:, :, -1, -2:] = 0
        beta[:, :, -1, -2:] = 0
        _, final, _ = chunked(q, k, v, g, beta, state)
        _, reference = recurrent(
            q.flatten(2, 3)[:, :, :-2],
            k.flatten(2, 3)[:, :, :-2],
            v.flatten(2, 3)[:, :, :-2],
            g.flatten(2, 3)[:, :, :-2],
            beta.flatten(2, 3)[:, :, :-2],
            state,
        )
        torch.testing.assert_close(final, reference, atol=2e-5, rtol=2e-5)

    def test_zero_beta_zero_decay_preserves_state(self):
        q, k, v, g, beta, state = self.inputs()
        g.zero_()
        beta.zero_()
        _, final, _ = chunked(q, k, v, g, beta, state)
        torch.testing.assert_close(final, state, atol=0, rtol=0)

    def test_query_scale_is_explicit_and_applied_once(self):
        q, k, v, g, beta, state = self.inputs()
        scaled, final, _ = chunked(q, k, v, g, beta, state)
        unscaled = q * 5**0.5
        output, other_final, _ = chunked(unscaled, k, v, g, beta, state, q_scale=5**-0.5)
        torch.testing.assert_close(output, scaled, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(other_final, final, atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
