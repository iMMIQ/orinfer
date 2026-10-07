import copy
import unittest

from tools.model.flash_next.config import CONTRACT, validate


def fixture(component='text'):
    text = copy.deepcopy(CONTRACT['supported_text'])
    text.update(max_position_embeddings=262144, rope_parameters=copy.deepcopy(CONTRACT['supported_rope']),
                mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False,
                mtp={'num_hidden_layers': 1, 'layer_types': ['full_attention'], 'hybrid': True,
                     'rope_theta': 10000000, 'mtp_use_hidden_state_from_layer': None})
    return {'model_type': 'qwen4_exp', 'text_config': text,
            'quantization_config': {'quant_method': 'orinfer_e8p_int8', 'version': 1,
                                    'basis': 'integer-e8p-spread29-v1', 'expert_rotation': 'signed-block128',
                                    'embedding_rotation': 'paley20-walsh8', 'compute_dtype': 'int8_quality',
                                    'component': component}}


class ConfigTests(unittest.TestCase):
    def test_text_and_shared_mtp_context(self):
        for component in ('text', 'mtp'):
            for capacity in (1, 512, 2048, 8192, 262144):
                validate(fixture(component), capacity)
        for capacity in (0, 262145, True, 512.0):
            with self.assertRaises(ValueError):
                validate(fixture(), capacity)
        config = fixture()
        config['text_config']['max_position_embeddings'] = 2048
        with self.assertRaises(ValueError):
            validate(config, 8192)

    def test_changed_architecture_and_routing_are_rejected(self):
        changes = {'hidden_size': 1280, 'num_experts_per_tok': 8, 'attention_bias': True,
                   'rms_norm_eps': 1e-5, 'ple_layer_ids': [3], 'indexer_compress_ratio': 8,
                   'linear_num_value_heads': 16, 'norm_topk_prob': False,
                   'rope_parameters': {'rope_theta': 10000}, 'layer_types': ['full_attention'] * 48}
        for key, value in changes.items():
            with self.subTest(key=key):
                config = fixture()
                config['text_config'][key] = value
                with self.assertRaises(ValueError):
                    validate(config, 512)
        for config in ({}, {'model_type': 'qwen4_exp', 'text_config': []}):
            with self.assertRaises(ValueError):
                validate(config, 512)

    def test_other_codecs_and_draft_semantics_are_rejected(self):
        for key, value in {'version': True, 'basis': 'other', 'compute_dtype': 'a4',
                           'expert_rotation': 'none', 'component': 'vision'}.items():
            config = fixture()
            config['quantization_config'][key] = value
            with self.assertRaises(ValueError):
                validate(config, 512)
        for field, value in {'mtp_use_dedicated_embeddings': True, 'mtp_num_hidden_layers': 2,
                             'mtp': {'layer_types': ['linear_attention']}}.items():
            config = fixture('mtp')
            config['text_config'][field] = value
            with self.assertRaises(ValueError):
                validate(config, 512)


if __name__ == '__main__':
    unittest.main()
