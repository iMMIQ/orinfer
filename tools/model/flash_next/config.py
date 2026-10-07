"""CPU-side contract for the currently implemented Flash Next Q2A8 recipe."""
import json
from pathlib import Path


CONTRACT = json.loads((Path(__file__).resolve().parents[3] /
                       'configs/architecture-contract.json').read_text())['families']['flash_next']


def validate(config, capacity):
    """Reject unsupported semantics before allocating weights or CUDA state."""
    if not isinstance(config, dict) or config.get('model_type') != 'qwen4_exp':
        raise ValueError('Flash Next qwen4_exp configuration required')
    text = config.get('text_config')
    if not isinstance(text, dict):
        raise ValueError('Flash Next text_config required')
    for key, expected in CONTRACT['supported_text'].items():
        actual = text.get(key)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f'Unsupported Flash Next text_config.{key}: {actual!r}')
    rope = text.get('rope_parameters', {})
    if not isinstance(rope, dict) or any(rope.get(k) != v for k, v in CONTRACT['supported_rope'].items()):
        raise ValueError('Unsupported Flash Next RoPE parameters')
    if text.get('norm_topk_prob', True) is not True:
        raise ValueError('Flash Next requires normalized top-k routing')
    maximum = text.get('max_position_embeddings')
    if (type(maximum) is not int or maximum < 1 or type(capacity) is not int
            or not 1 <= capacity <= min(maximum, 262144)):
        raise ValueError('Context exceeds checkpoint/native 262144 limit')
    quant = config.get('quantization_config')
    if not isinstance(quant, dict):
        raise ValueError('Flash Next quantization_config required')
    for key, value in {'quant_method': 'orinfer_e8p_int8', 'version': 1,
                       'basis': 'integer-e8p-spread29-v1', 'expert_rotation': 'signed-block128',
                       'embedding_rotation': 'paley20-walsh8', 'compute_dtype': 'int8_quality'}.items():
        if quant.get(key) != value or type(quant.get(key)) is not type(value):
            raise ValueError(f'Unsupported Flash Next quantization_config.{key}')
    component = quant.get('component', 'text')
    if component not in ('text', 'mtp'):
        raise ValueError('Unsupported Flash Next checkpoint component')
    if component == 'mtp':
        mtp = text.get('mtp', {})
        if (not isinstance(mtp, dict) or text.get('mtp_num_hidden_layers') != 1
                or mtp.get('num_hidden_layers') != 1 or mtp.get('layer_types') != ['full_attention']
                or mtp.get('rope_theta') != 10000000 or mtp.get('hybrid') is not True
                or mtp.get('mtp_use_hidden_state_from_layer') is not None
                or text.get('mtp_use_dedicated_embeddings') is not False):
            raise ValueError('Only the shared one-layer Flash Next MTP is supported')
