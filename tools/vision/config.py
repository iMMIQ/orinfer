"""Validate assumptions of the Qwen3.5 image adapter before adopting weights."""
import math


def validate_adapter(config, manifest):
    vision, text = config['vision_config'], config['text_config']
    validate_vision(vision, 5120)
    rope = text['rope_parameters']
    if ((text['hidden_size'], text['num_attention_heads'], text['num_key_value_heads'],
         text['head_dim']) != (5120, 24, 4, 256)
            or text['vocab_size'] != manifest['vocab']
            or not 0 < manifest['max_context'] <= 8704
            or rope.get('rope_type') != 'default'
            or rope.get('rope_theta') != 10_000_000
            or rope.get('partial_rotary_factor') != .25
            or rope.get('mrope_section') != [11, 11, 10]
            or rope.get('mrope_interleaved') is not True):
        raise ValueError('Checkpoint/text bridge dimensions or rotary layout mismatch')
    buffers = {b['name']: b for b in manifest['buffers']}
    vocab = manifest['vocab']
    expected = {
        'Embedding_P': ('u8', [vocab, 2560]),
        'Embedding_S': ('f16', [vocab, 40]),
        'Embedding_Z': ('i8', [vocab, 40]),
        'FullX': ('f16', [manifest['chunk_tokens'], 14336]),
        'Hidden': ('f16', [manifest['chunk_tokens'], 5120]),
    }
    for name, (dtype, shape) in expected.items():
        b = buffers.get(name)
        if b is None or (b['dtype'], b['shape']) != (dtype, shape):
            raise ValueError(f'Text plan requires compatible group-128 W4 bridge buffer {name}')
    rotary = buffers.get('Rotary')
    if (rotary is None or rotary['dtype'] != 'f16'
            or rotary['shape'] != [manifest['max_context'], 64]):
        raise ValueError('Text plan requires a compatible rotary cache')


def validate_vision(vision, output_hidden):
    if (vision['hidden_size'], vision['num_heads'], vision['in_channels'],
        vision['patch_size'], vision['spatial_merge_size'], vision['temporal_patch_size'],
        vision['out_hidden_size'], vision['hidden_act']) != (
            1152, 16, 3, 16, 2, 2, output_hidden, 'gelu_pytorch_tanh'):
        raise ValueError('Unsupported vision architecture; need an explicit adapter')
    positions = vision['num_position_embeddings']
    if (vision.get('deepstack_visual_indexes') or vision['depth'] <= 0
            or vision['intermediate_size'] <= 0 or positions <= 0
            or math.isqrt(positions)**2 != positions):
        raise ValueError('Unsupported vision positions, depth or DeepStack layout')
