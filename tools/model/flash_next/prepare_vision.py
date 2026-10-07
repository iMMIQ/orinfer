"""Attach shared ViT, Flash MRoPE and MTP visual bridges to a native package.

Immutable text weights/cubins are reused without loading the complete text model.
The registered Rust recipe validates every exported binding before publication.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import shutil

from kernels.model import flash_next, qsa
from kernels.vision.features import overlay
from tools.model.flash_next.prepare import Publisher
from tools.model.publication import link_or_copy, write_json, seed_compile_cache
from tools.operators.common import configure
from tools.operators.abi import parse_host, evaluate
from tools.vision.plan import EncoderBuilder
from tools.vision.config import validate_vision
from tools.model.flash_next.config import CONTRACT


class VisionPublisher(EncoderBuilder):
    def __init__(self, model, checkpoint, max_patches, max_features):
        self.root = model
        self.output = model / 'cache'
        self.build = json.loads((self.output / 'build.json').read_text())
        self.manifest = self.build['metadata']
        self.manifest['kernels'] = self.build['kernels']
        self.manifest['programs'] = self.build['programs']
        self.config = json.loads((model / 'config.json').read_text())
        self.vision = self.config['vision_config']
        validate_vision(self.vision, 2560)
        if any(self.vision.get(k) != v for k, v in CONTRACT['supported_vision'].items()):
            raise ValueError('Unsupported Flash Next vision configuration')
        rope = self.config['text_config']['rope_parameters']
        if rope.get('mrope_interleaved') is not True or rope.get('mrope_section') != [11,11,10]:
            raise ValueError('Unsupported Flash Next interleaved MRoPE')
        if self.manifest.get('vision'):
            raise ValueError('Source already contains a vision encoder')
        if not max_patches // 4 <= max_features <= self.manifest['max_context']:
            raise ValueError('Invalid multi-image feature budget')
        self.max_patches, self.max_features = max_patches, max_features
        self.source = checkpoint / 'model.safetensors'
        shutil.copyfile(checkpoint / 'preprocessor_config.json', model / 'preprocessor_config.json')
        original = json.loads((checkpoint / 'source.json').read_text())
        quant = self.config['quantization_config']
        if (original['repo'], original['revision']) != (quant['source'], quant['source_revision']):
            raise ValueError('Vision and language weights must share the pinned checkpoint')
        from tools.model.publication import file_hash
        if file_hash(self.source) != original['sha256']:
            raise ValueError('Original vision source hash mismatch')
        self.dtype, self.kernel_dtype = 'f16', 'float16'
        self.fp32_interpolation = True
        self.exports = {}
        self.buffers = {b['name']: b for b in self.manifest['buffers']}
        self.publisher = Publisher(None, model)
        self.publisher.buffers = self.buffers
        self.publisher.scopes = self.build['scopes']
        self.publisher.weight_map = json.loads((self.output / 'weights/model.safetensors.index.json').read_text())['weight_map']

    def buffer(self, name, shape, dtype='f16', data=None, weight=False):
        if name in self.buffers:
            raise ValueError(f'Vision buffer collides with text: {name}')
        if data is not None:
            self.publisher.buffer(data, name, 'weights')
            # Tensor views are consumed immediately; allocator addresses may be reused.
            self.publisher.storage.clear()
        else:
            scope = 'sequence' if name in ('Features', 'FeatureIndex', 'MRopePositions', 'MtpFeatureIndex') else 'workspace'
            self.publisher.fixed(name, dtype, list(shape), scope)

    def compile(self, name, factory, *, bf16=True):
        dtype = self.dtype
        if not bf16:
            self.dtype = 'f16'
        try:
            super().compile(name, factory)
        finally:
            self.dtype = dtype
        # Assets must travel inside the execution package's kernel directory.
        for field in ('module', 'source', 'host_abi'):
            asset = self.exports[name][field]
            source = self.output / asset['file']
            destination = self.output / 'kernels' / ('vision-' + name) / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(source, destination)
            asset['file'] = str(destination.relative_to(self.output))

    def emit(self, program, name, rows, **pointers):
        # The ViT order is shared with Qwen3.5; expose registered section slots.
        patches = rows * 4 if name.startswith('merge') else rows
        count = len(program)
        if count < 2:
            section, slot = 'begin', count
        elif count < 2 + 27 * 10:
            section, slot = f'layer{(count-2)//10}', (count-2) % 10
        else:
            section, slot = 'end', count - 2 - 27 * 10
        k = self.kernel(name, pointers, rows, f'vision_m{patches}/{section}/k{slot}')
        self.manifest['kernels'].append(k)
        program.append(dict(kind='kernel', name=k['name']))

    def binding(self, factory, pointers, rows, name):
        abi = self.exports[factory]
        scalars = {'rows': rows, 'm': rows}
        args = []
        for arg in abi['ordered_arguments']:
            expression, dtype = arg['value'], arg['ctype']
            if dtype in ('ctypes.c_void_p', 'c_void_p'):
                args.append(copy.deepcopy(pointers[expression.removesuffix('.data_ptr()')]))
            else:
                if dtype not in ('ctypes.c_int', 'ctypes.c_int32'):
                    raise ValueError('Unsupported visual kernel scalar ABI')
                args.append(dict(kind='i32', value=evaluate(expression, scalars)))
        launch = abi['launch_expressions']
        return dict(name=name, **{key: abi[key] for key in ('module','source','host_abi','symbol')},
                    grid=[evaluate(launch['gridDim'+a], scalars) for a in 'XYZ'],
                    block=[evaluate(launch['blockDim'+a], scalars) for a in 'XYZ'],
                    shared_memory_bytes=evaluate(launch['sharedMemBytes'], scalars), cooperative=False, args=args)

    def bridges(self):
        m = self.manifest
        capacity = m['max_context']
        if m.get('mtp'):
            self.buffer('MtpFeatureIndex', [capacity], 'i32')
            m['mtp']['feature_index'] = 'MtpFeatureIndex'
        counts = {'qsa-prepare': 0, 'index-query': 0, 'index-compress': 0}
        for index, kernel in enumerate(m['kernels']):
            old_factory = Path(kernel['module']['file']).parent.name
            match = re.fullmatch(r'(qsa-prepare|index-query|index-compress)-(\d+)', old_factory)
            if not match:
                continue
            kind, width = match[1], int(match[2])
            factory = f'{kind}-mrope-{width}'
            if factory not in self.exports:
                build = (lambda: flash_next.qsa_prepare(width, capacity, is_neox_style=True, staged=True, mrope=True)) if kind == 'qsa-prepare' else (
                        lambda: qsa.index_query(width, capacity)) if kind == 'index-query' else (
                        lambda: qsa.index_compress(width, capacity, mrope=True))
                self.compile(factory, build, bf16=False)
            old_abi, = parse_host((self.output / kernel['host_abi']['file']).read_text())
            pointers = {a['value'].removesuffix('.data_ptr()'): value for a, value in zip(old_abi['ordered_arguments'], kernel['args'])
                        if a['ctype'] in ('ctypes.c_void_p', 'c_void_p')}
            pointers['Coordinates'] = dict(kind='buffer', name='MRopePositions')
            m['kernels'][index] = self.binding(factory, pointers, width, kernel['name'])
            counts[kind] += 1
        if min(counts.values()) == 0:
            raise ValueError('Missing Flash QSA/indexer bridge bindings')
        for program, groups in list(self.build['groups'].items()):
            if not (program == 'decode' or program.startswith(('prefill_m', 'verify_m', 'mtp_warm_m'))):
                continue
            width = 1 if program == 'decode' else int(program.rsplit('_m', 1)[1])
            draft = program.startswith('mtp_warm')
            insert = 2 if draft else 0
            beginning = groups['begin']
            renames = {op['name']: f'{program}/begin/k{i+1}' for i, op in enumerate(beginning) if i >= insert}
            for kernel in m['kernels']:
                kernel['name'] = renames.get(kernel['name'], kernel['name'])
            for ops in [*m['programs'].values(), *(g for sections in self.build['groups'].values() for g in sections.values())]:
                for op in ops:
                    if op['kind'] == 'kernel':
                        op['name'] = renames.get(op['name'], op['name'])
            factory = f'overlay-{width}'
            if factory not in self.exports:
                self.compile(factory, lambda: overlay(width, 2560, capacity, self.max_features), bf16=False)
            buffers = dict(Embedding=f'{"Draft" if draft else ""}M{width}_Embedding', Features='Features',
                           Index='MtpFeatureIndex' if draft else 'FeatureIndex', Position='MtpPosition' if draft else 'Position')
            kernel = self.binding(factory, {k: dict(kind='buffer', name=v) for k, v in buffers.items()}, width, f'{program}/begin/k{insert}')
            m['kernels'].append(kernel)
            op = dict(kind='kernel', name=kernel['name'])
            beginning.insert(insert, op)
            m['programs'][program].insert(insert, copy.deepcopy(op))
        if m.get('mtp'):
            m['programs']['mtp_draft'] = (m['programs']['mtp_warm_m1'][1:] + m['programs']['mtp_head_m1'])
        print('FLASH VISUAL BRIDGES', counts, flush=True)

    def finish(self):
        parameters = self.build_encoder()
        self.bridges()
        m = self.manifest
        m['buffers'] = list(self.buffers.values())
        m['weight_parameters'] += parameters
        sizes = {'u8':1,'i8':1,'u16':2,'f16':2,'bf16':2,'f32':4,'u32':4,'i32':4,'i64':8}
        m['weight_bytes'] = sum(math.prod(b['shape'])*sizes[b['dtype']] for b in m['buffers'] if b['access'] == 'read')
        m['weight_scope'] += '; original vision weights in FP16, images and multiple images'
        self.config['language_model_only'] = False
        write_json(self.root / 'config.json', self.config)
        write_json(self.output / 'weights/model.safetensors.index.json', dict(metadata=dict(total_size=m['weight_bytes']), weight_map=self.publisher.weight_map))
        self.build['kernels'] = m.pop('kernels')
        self.build['programs'] = m.pop('programs')
        write_json(self.output / 'build.json', self.build)
        print('FLASH VISION BUILD COMPLETE', parameters, m['weight_bytes'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model-output', type=Path, required=True)
    p.add_argument('--max-patches', type=int, choices=(1024,2048,4096,8192,16384,32768), default=8192)
    p.add_argument('--max-features', type=int, default=16384)
    p.add_argument('--compile-cache', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); configure()
    if a.compile_cache:
        seed_compile_cache(a.compile_cache, a.output/'cache/0.1.15')
    if a.model_output.exists():
        raise FileExistsError(a.model_output)
    if a.model_output.resolve().is_relative_to(a.base_model.resolve()):
        raise ValueError('Output must be outside the source model')
    def clone(source, destination):
        if Path(source).suffix in ('.safetensors','.cubin','.cu','.txt'):
            link_or_copy(source, destination)
        else:
            shutil.copyfile(source, destination)
    shutil.copytree(a.base_model, a.model_output, copy_function=clone, ignore=shutil.ignore_patterns('packages','aot'))
    VisionPublisher(a.model_output, a.checkpoint, a.max_patches, a.max_features).finish()


if __name__ == '__main__': main()
