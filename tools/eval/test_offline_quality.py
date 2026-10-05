"""Scoring must reject mismatched histories and retain actual execution modes."""
import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.eval.offline_quality import add_queries, compare, prepare
from tools.eval.scoring_common import SEED, context_hash


def reports():
    requests = dict(seed=SEED,cases=[dict(id='case',prompt_ids=[10,11],target_ids=[3],query_ids=[])])
    probe = dict(case_id='case',execution_mode='prefill',position=0,seed=SEED,
                 context_sha256=context_hash([10,11]),reference_token_id=3,reference_logprob=-0.1,
                 top3=[dict(token_id=3,logprob=-0.1),dict(token_id=4,logprob=-3.),dict(token_id=5,logprob=-4.)],
                 queried_logprobs={'3':-0.1,'4':-3.,'5':-4.})
    baseline = dict(seed=SEED,complete=True,revision='source-revision',execution='full-sequence prefill',probes=[probe])
    candidate = dict(seed=SEED,manifest_sha256='candidate',probes=[dict(copy.deepcopy(probe),execution_mode='decode')])
    return requests,baseline,candidate


class OfflineQualityTests(unittest.TestCase):
    def test_image_hash_and_pairing_reject_changed_pixels_and_image_order(self):
        image = dict(grid_height=2, grid_width=2, pixels=[0.] * 6144)
        self.assertEqual(context_hash([1, 2], [image]),
                         'bf01648ff97c3d303d1e215026c2a49fa1b829b8ee66fb1edd201046f8ca8977')
        changed = copy.deepcopy(image)
        changed['pixels'][0] = 1.
        self.assertNotEqual(context_hash([1, 2], [image, changed]),
                            context_hash([1, 2], [changed, image]))
        for images in ([image], [image, changed]):
            requests, baseline, candidate = reports()
            requests['cases'][0]['images'] = copy.deepcopy(images)
            for report in (baseline, candidate):
                report['probes'][0]['context_sha256'] = context_hash([10, 11], images)
            self.assertEqual(compare(requests, baseline, candidate)['aggregate']['delta_target_nll'], 0.)
            requests['cases'][0]['images'][0]['pixels'][0] = .5
            with self.assertRaisesRegex(ValueError, 'different teacher-forced history'):
                compare(requests, baseline, candidate)

    def test_explicit_reference_queries_and_cross_path_pairing(self):
        requests,baseline,candidate = reports()
        self.assertEqual(add_queries(requests,baseline)['cases'][0]['query_ids'],[[3,4,5]])
        result = compare(requests,baseline,candidate)
        self.assertEqual(result['aggregate']['delta_target_nll'],0.)
        self.assertEqual(result['paired'][0]['execution_mode'],'decode_vs_bf16_prefill')
        self.assertEqual(baseline['probes'][0]['execution_mode'],'prefill')
        self.assertEqual(candidate['probes'][0]['execution_mode'],'decode')

    def test_incomplete_reference_wrong_history_and_missing_query_are_rejected(self):
        for mutation in ('incomplete','history','query','extra_case'):
            requests,baseline,candidate = reports()
            if mutation=='incomplete':baseline['complete']=False
            elif mutation=='history':candidate['probes'][0]['context_sha256']=context_hash([10,12])
            elif mutation=='query':del candidate['probes'][0]['queried_logprobs']['5']
            else:candidate['probes'].append(dict(candidate['probes'][0],case_id='extra'))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                compare(requests,baseline,candidate)

    def test_template_mapping_produces_json_serializable_histories(self):
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:{'input_ids':[10,11]},
                                  encode=lambda *a,**k:[3])
        transformers=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tokenizer))
        fixtures=dict(seed=SEED,cases=[dict(id='case',messages=[],target_text='valid answer')])
        with patch.dict(sys.modules,{'transformers':transformers}):
            self.assertEqual(prepare('local',fixtures)['cases'][0]['prompt_ids'],[10,11])
            del fixtures['cases'][0]['target_text']
            with self.assertRaisesRegex(ValueError,'target_text'):
                prepare('local',fixtures)


if __name__=='__main__':
    unittest.main()
