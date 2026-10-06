"""Bounded CPU reads of our published Flash Next weight representation.

No community decoder or conversion progress file is needed. The standard HF
index selects shards; safetensors metadata specifies the sliced physical layout.
This is an offline reference reader, not the online Rust architecture adapter.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from tools.model.flash_weights import header
from tools.model.safetensors_source import relative_name, SIZES
from tools.quantization.embedding_vq import EmbeddingDecoder
from tools.quantization.flash_next import digest


DTYPES = {'F16':'<f2','F32':'<f4','F64':'<f8','I8':'i1','U8':'u1','I16':'<i2','U16':'<u2',
          'I32':'<i4','U32':'<u4','I64':'<i8','U64':'<u8','BOOL':'?'}


class Checkpoint:
    def __init__(self, directory, *, verify_hashes=True):
        self.directory = Path(directory)
        self.embedding_decoder = EmbeddingDecoder()
        self.config = json.loads((self.directory/'config.json').read_text())
        if self.config.get('quantization_config',{}).get('quant_method') != 'orinfer_e8p_int8':
            raise ValueError('Our Flash Next quantization config is required')
        index = json.loads((self.directory/'model.safetensors.index.json').read_text())
        self.parts = defaultdict(list)
        hashes = index.get('metadata',{}).get('orinfer.sha256',{})
        observed = {};payload_bytes = 0
        for filename in sorted(set(index['weight_map'].values())):
            relative_name(filename)
            path = self.directory/filename
            if verify_hashes and (filename not in hashes or digest(path) != hashes[filename]):
                raise ValueError('Checkpoint shard identity mismatch')
            begin,data = header(path);meta = data.get('__metadata__',{})
            if meta.get('format') != 'orinfer.flash_next.weights.v1':raise ValueError('Unsupported physical weight format')
            for key,t in data.items():
                if key == '__metadata__':continue
                if key in observed:raise ValueError('Duplicate indexed tensor')
                observed[key] = filename
                payload_bytes += t['data_offsets'][1]-t['data_offsets'][0]
            keys = json.loads(meta['keys']);shape = json.loads(meta['source_shape'])
            if set(keys.values()) != set(data)-{'__metadata__'}:raise ValueError('Physical key map mismatch')
            part = {'path':path,'begin':begin,'header':data,'keys':keys,'shape':shape,
                    'kind':meta['kind'],'first':int(meta['first']),'count':int(meta['count'])}
            self.parts[meta['source_tensor']].append(part)
        if observed != index['weight_map'] or payload_bytes != index.get('metadata',{}).get('total_size'):
            raise ValueError('Standard safetensors index/payload disagreement')
        for name,parts in self.parts.items():
            parts.sort(key=lambda p:p['first'])
            shape = parts[0]['shape'];kind = parts[0]['kind'];cursor = 0
            for p in parts:
                if p['first'] != cursor or p['count'] <= 0 or p['shape'] != shape or p['kind'] != kind:
                    raise ValueError(f'Incomplete/overlapping logical tensor: {name}')
                cursor += p['count']
            if cursor != (shape[0] if shape else 1):raise ValueError(f'Truncated logical tensor: {name}')

    def shape(self, name):
        return tuple(self.parts[name][0]['shape'])

    def kind(self, name):
        return self.parts[name][0]['kind']

    def rows(self, name, start, count):
        parts = self.parts[name];shape = self.shape(name)
        if not shape or type(start) is not int or type(count) is not int or start < 0 or count <= 0 or start+count > shape[0]:
            raise ValueError('Invalid logical row range')
        result = []
        for p in parts:
            left = max(start,p['first']);right = min(start+count,p['first']+p['count'])
            if right <= left:continue
            first,last = left-p['first'],right-p['first'];kind = p['kind'];keys = p['keys']
            if kind == 'e8p-expert':raise ValueError('Use expert_parts for packed routed expert banks')
            if kind == 'original':
                info = p['header'][keys[name]];width = int(np.prod(shape[1:],dtype=np.int64))
                row_bytes = width*SIZES[info['dtype']]
                with p['path'].open('rb') as f:
                    f.seek(p['begin']+info['data_offsets'][0]+first*row_bytes)
                    raw = f.read((last-first)*row_bytes)
                if len(raw) != (last-first)*row_bytes:raise EOFError('Truncated original-precision rows')
                if info['dtype'] == 'BF16':
                    value = (np.frombuffer(raw,dtype='<u2').astype(np.uint32) << 16).view(np.float32)
                else:value = np.frombuffer(raw,dtype=DTYPES[info['dtype']]).copy()
                value = value.reshape(right-left,*shape[1:])
            else:
                with safe_open(p['path'],framework='np') as f:
                    if kind == 'int8-row':
                        value = f.get_slice(keys['weight'])[first:last].astype(np.float32)
                        value *= f.get_slice(keys['scale'])[first:last].astype(np.float32)[:,None]
                    elif kind == 'e8p-embedding':
                        value = self.embedding_decoder.decode(f.get_slice(keys['rows'])[first:last],
                                                              f.get_tensor(keys['table']),f.get_tensor(keys['signs']))
                    else:raise ValueError('Unsupported quantized row kind')
            result.append(value)
        return np.concatenate(result,axis=0)

    def tensor(self, name):
        shape = self.shape(name)
        if shape:return self.rows(name,0,shape[0])
        p = self.parts[name][0];info = p['header'][p['keys'][name]]
        with p['path'].open('rb') as f:
            f.seek(p['begin']+info['data_offsets'][0]);raw = f.read(SIZES[info['dtype']])
        if info['dtype'] == 'BF16':return (np.frombuffer(raw,'<u2').astype(np.uint32) << 16).view(np.float32).reshape(())
        return np.frombuffer(raw,DTYPES[info['dtype']]).reshape(())

    def expert_parts(self, name):
        if self.kind(name) != 'e8p-expert':raise ValueError('Expected a routed expert bank')
        for p in self.parts[name]:
            with safe_open(p['path'],framework='np') as f:
                yield p['first'],{key:f.get_tensor(value) for key,value in p['keys'].items()}

    def int8_parts(self, name):
        if self.kind(name) != 'int8-row':raise ValueError('Expected a dense INT8 matrix')
        for p in self.parts[name]:
            with safe_open(p['path'],framework='np') as f:
                yield p['first'],f.get_tensor(p['keys']['weight']),f.get_tensor(p['keys']['scale'])
