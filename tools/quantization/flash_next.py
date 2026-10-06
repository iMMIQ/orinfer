"""Resumable original-BF16 expert conversion to our integer E8P + INT8 layout.

This stage converts all routed experts. PLE and the remaining text tensors are
separate stages; the manifest never declares the entire model complete here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from tools.model.safetensors_source import Source


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024**2),b''):h.update(data)
    return h.hexdigest()


def equivalent(left, right):
    """Ignore metadata key ordering while comparing exact tensor bytes."""
    def identity(path):
        h = hashlib.sha256()
        with open(path,'rb') as f:
            size, = struct.unpack('<Q',f.read(8))
            if size > 64*1024**2:raise ValueError('Invalid quantized shard header')
            header = json.loads(f.read(size))
            for raw in iter(lambda:f.read(4*1024**2),b''):h.update(raw)
        return header,h.hexdigest()
    return identity(left) == identity(right)


def bf16(data, shape):
    bits = np.frombuffer(data,dtype='<u2').astype(np.uint32) << 16
    return bits.view(np.float32).reshape(shape)


def task_list(source, *, layers, chunk_experts, component='text'):
    if component not in ('text','mtp'):raise ValueError('Invalid Flash component')
    tasks = []
    for layer in layers:
        for family in ('gate_up','down'):
            prefix = 'model.language_model' if component == 'text' else 'mtp'
            name = f'{prefix}.layers.{layer}.mlp.experts.{family}_proj'
            filename,begin,t = source.tensor(name)
            if t['dtype'] != 'BF16' or len(t['shape']) != 3 or t['shape'][0] != 512 or t['shape'][2]%128:
                raise ValueError(f'Unexpected original expert bank: {name}')
            e,n,k = t['shape']
            for start in range(0,e,chunk_experts):
                count = min(chunk_experts,e-start)
                tasks.append({'tensor':name,'shape':[count,n,k],'first_expert':start,
                              'source_file':filename,'filename':f'layer-{layer:02d}-{family}-{start:03d}.safetensors'})
    return sorted(tasks,key=lambda t:(int(t['filename'].split('-')[1]),t['first_expert'],
                                     0 if 'gate_up' in t['filename'] else 1))


def verify(path, task, record, contract):
    if not path.is_file() or digest(path) != record.get('sha256'):
        raise ValueError(f'Incomplete/corrupt quantized shard: {path.name}')
    e,n,k = task['shape']
    with safe_open(path,framework='np') as f:
        if set(f.keys()) != {'indices','table','scales','signs'} or (f.metadata() or {}).get('contract') != contract:
            raise ValueError('Quantized shard contract mismatch')
        for key,shape,dtype in [('indices',(e,k//128,n,16),np.uint16),('table',(256,8),np.int8),
                                ('scales',(e,n),np.float16),('signs',(k,),np.int8)]:
            array = f.get_tensor(key)
            if array.shape != shape or array.dtype != dtype:raise ValueError('Quantized tensor shape/dtype mismatch')
        scales = f.get_tensor('scales')
        if not np.isfinite(scales).all() or (scales <= 0).any():raise ValueError('Invalid stored scales')
        if not np.isin(f.get_tensor('signs'),[-1,1]).all():raise ValueError('Invalid stored rotation')
        if 'tensor' in task:
            metadata = f.metadata()
            if metadata.get('source_tensor') != task['tensor'] or metadata.get('first_expert') != str(task['first_expert']):
                raise ValueError('Shard source/expert span mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--repo',default='Qwen/Qwen3.8-Flash-Next')
    p.add_argument('--revision',required=True)
    p.add_argument('--source-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',default='0:48')
    p.add_argument('--chunk-experts',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261002)
    p.add_argument('--max-chunks',type=int)
    p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--stage',choices=('experts','aux','all'),default='experts')
    p.add_argument('--component',choices=('text','mtp'),default='text')
    p.add_argument('--aux-kind',choices=('original','int8-row','e8p-embedding'))
    a = p.parse_args()
    if not re.fullmatch(r'\d+:\d+',a.layers):p.error('--layers must be start:end')
    first,last = map(int,a.layers.split(':'))
    if not 0 <= first < last <= (48 if a.component == 'text' else 1) or not 1 <= a.chunk_experts <= 32 or not 1 <= a.workers <= 4:
        p.error('Invalid layer range, chunk size or worker count')
    if a.max_chunks is not None and a.max_chunks <= 0:p.error('--max-chunks must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    source = Source(a.index,directory=a.source_dir,repo=a.repo,revision=a.revision,cache=a.output/'source-headers')
    # Version pins basis construction, input rotation, row-scale fitting and
    # physical group-major layout. Changing any part invalidates resume.
    contract = json.dumps({'format':'orinfer.flash_next.experts.v1','source':a.repo,'revision':a.revision,
                           'source_index_sha256':digest(a.index),'seed':a.seed,'layers':[first,last],
                           'chunk_experts':a.chunk_experts,'basis':'integer-e8p-spread29-v1',
                           'rotation':'signed-block128','fit':'weight-only-ls2','layout':'expert-group-row-vector'},sort_keys=True)
    if a.component == 'mtp':contract=json.dumps(dict(json.loads(contract),component='mtp'),sort_keys=True)
    state_path = a.output/'experts-progress.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['contract'] != contract:raise ValueError('Existing conversion uses a different contract')
    else:
        state = {'contract':contract,'shards':{},'routed_experts_complete':False,'model_complete':False,
                 'remaining_model_stages':(['non-expert MTP weights','end-to-end MTP validation'] if a.component=='mtp' else
                                           ['non-expert text weights','PLE embedding','end-to-end quality validation'])}
    print('Reading original expert bank headers',flush=True)
    tasks = task_list(source,layers=range(first,last),chunk_experts=a.chunk_experts,component=a.component)
    state['expected_shards'] = len(tasks)
    if a.inventory_only:
        state['source_expert_bytes'] = sum(np.prod(t['shape']).item()*2 for t in tasks)
        atomic_json(state_path,state);return
    pending = []
    for task in tasks:
        path = a.output/task['filename']
        if task['filename'] in state['shards']:
            verify(path,task,state['shards'][task['filename']],contract)
        else:
            if path.exists():
                # The artifact may have been renamed immediately before a
                # crash. Recompute from its pinned source and compare; do not
                # trust an unrecorded artifact as completed.
                path.rename(path.with_suffix('.unrecorded'))
            pending.append(task)
    if a.max_chunks:pending = pending[:a.max_chunks]
    from tools.operators.common import configure
    from tools.quantization.e8p_gpu import Encoder
    configure();enc = Encoder()
    if a.stage == 'aux':
        from tools.quantization.flash_next_aux import convert
        convert(source,a.output,enc,source_contract=contract,seed=a.seed,max_chunks=a.max_chunks,kind_filter=a.aux_kind,component=a.component)
        return
    sign_cache = {}
    def fetch(task):
        raw = source.rows(task['tensor'],task['first_expert'],task['shape'][0])
        return raw,hashlib.sha256(raw).hexdigest()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        # Submit only a bounded window; executor.map would enqueue the model.
        futures = {i:pool.submit(fetch,t) for i,t in enumerate(pending[:a.workers])}
        for i,task in enumerate(pending):
            raw,source_hash = futures.pop(i).result()
            j = i+a.workers
            if j < len(pending):futures[j] = pool.submit(fetch,pending[j])
            e,n,k = task['shape']
            if task['tensor'] not in sign_cache:
                tensor_seed = int.from_bytes(hashlib.sha256(f'{a.seed}:{task["tensor"]}'.encode()).digest()[:8],'little')
                sign_cache[task['tensor']] = np.random.default_rng(tensor_seed).choice(np.array([-1,1],np.int8),k)
            floating = bf16(raw,(e*n,k));del raw
            fit_started = time.perf_counter()
            w = enc.fit(floating,sign_cache[task['tensor']])
            target = w.indices.reshape(e,n,k//128,16).transpose(0,2,1,3).copy()
            path = a.output/task['filename'];temporary = path.with_suffix('.tmp')
            tensors = {'indices':target,'table':w.table,'scales':w.scales.reshape(e,n),'signs':w.signs}
            save_file(tensors,str(temporary),metadata={'contract':contract,'source_tensor':task['tensor'],
                       'logical_shape':json.dumps(task['shape']),'first_expert':str(task['first_expert']),
                       'source_range_sha256':source_hash})
            with temporary.open('rb') as f:os.fsync(f.fileno())
            temporary.replace(path)
            unrecorded = path.with_suffix('.unrecorded')
            if unrecorded.exists():
                if not equivalent(unrecorded,path):raise ValueError('Interrupted artifact differs from recomputation')
                unrecorded.unlink()
            record = dict(task,sha256=digest(path),bytes=path.stat().st_size,source_range_sha256=source_hash,
                          fit_seconds=time.perf_counter()-fit_started)
            verify(path,task,record,contract)
            state['shards'][task['filename']] = record
            state['elapsed_seconds_this_run'] = time.perf_counter()-started
            state['routed_experts_complete'] = first == 0 and last == (48 if a.component == 'text' else 1) and len(state['shards']) == len(tasks)
            atomic_json(state_path,state)
            print(json.dumps({'done':len(state['shards']),'total':len(tasks),'shard':task['filename'],
                              'fit_seconds':record['fit_seconds'],'bytes':record['bytes']}),flush=True)
            del floating,w,target,tensors
    if a.stage == 'all' and state['routed_experts_complete']:
        from tools.quantization.flash_next_aux import convert
        auxiliary = convert(source,a.output,enc,source_contract=contract,seed=a.seed,max_chunks=a.max_chunks,kind_filter=a.aux_kind,component=a.component)
        state['weights_complete'] = auxiliary['complete']
        if auxiliary['complete']:state['remaining_model_stages'] = ['end-to-end quality validation']
        atomic_json(state_path,state)


if __name__ == '__main__':main()
