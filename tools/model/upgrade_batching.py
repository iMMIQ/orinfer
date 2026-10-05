"""Offline batch operator package and 32/64/128-token mixed-prefill profiles.

Resident weights are hardlinked unchanged. Dynamic-row projection cubins serve
all batch buckets; no Python compiler runs on the online request path.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model.publication import atomic_model, clone_model, commit_package, file_hash, load_model, source_path, write_json
from tools.operators.abi import parse_host, evaluate


def bind(export, name, pointers, dimensions):
    args = []
    for arg in export['ordered_arguments']:
        if arg['ctype'] in ('ctypes.c_void_p', 'c_void_p'):
            variable = arg['value'].removesuffix('.data_ptr()')
            args.append(dict(kind='buffer', name=pointers[variable]))
        elif arg['ctype'] in ('ctypes.c_int32', 'c_int32'):
            args.append(dict(kind='i32', value=evaluate(arg['value'], dimensions)))
        else:
            raise ValueError(f'Unsupported batch ABI: {arg}')
    launch = export['launch_expressions']
    return dict(name=name, module=export['module'], source=export['source'],
                host_abi=export['host_abi'], symbol=export['symbol'], args=args,
                grid=[evaluate(launch['gridDim' + axis], dimensions) for axis in 'XYZ'],
                block=[evaluate(launch['blockDim' + axis], dimensions) for axis in 'XYZ'],
                shared_memory_bytes=evaluate(launch['sharedMemBytes'], dimensions), cooperative=False)


def upgrade(model, destination, report):
    # Imported only inside the offline CUDA compiler container.
    import tilelang.language as T
    from tools.operators.common import configure, export_kernel
    from kernels.model.w4_small_m import w4_small_m
    from kernels.model.gdn_sequence import gdn_sequence
    from kernels.model import control
    from kernels.model.kv_int8 import attention_prefill_int8
    from kernels.operators.op02_residual_norm import residual_norm
    from kernels.operators.op07_gdn_ab import gdn_ab_tensorcore
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.operators.op23_final_norm import final_norm
    from kernels.operators.op32_split_k_merge import split_k_merge

    configure()
    model = model.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise ValueError('Destination already exists')
    data, origin, package = load_model(model)
    metadata = data['metadata']
    if package.get('batch_profiles'):
        raise ValueError('Model already has batch operators')
    if not metadata.get('mtp') or not metadata.get('kv_cache', {}).get('buffers'):
        raise ValueError('Batch upgrade requires native MTP and direct INT8 KV profiles')
    cache = destination / 'cache'
    operator = clone_model(model, destination, origin)
    # Replace hardlinked package metadata atomically after compiling.
    kernels = {k['name']: k for k in package['kernels']}
    buffers = {b['name']: b for b in metadata['buffers']}
    config = json.loads((model / 'config.json').read_text())
    text = config.get('text_config', config)
    h, f, vocab = text['hidden_size'], text['intermediate_size'], metadata['vocab']
    if (h, f) != (5120, 17408):
        raise ValueError('Batch projection factories currently support 27B dimensions')
    exports = {}

    def export_existing(key, kernel):
        host = parse_host(source_path(operator, kernel['host_abi']['file']).read_text())
        if len(host) != 1 or host[0]['symbol'] != kernel['symbol']:
            raise ValueError('Expected one verified host export')
        exports[key] = {**host[0], **{k: kernel[k] for k in ('module', 'source', 'host_abi')}}

    def compile_kernel(key, factory):
        print('compile', key, flush=True)
        out = operator / 'batch-aot' / key
        export_kernel(factory(), out)
        host = parse_host((out / 'host.txt').read_text())
        if len(host) != 1:
            raise ValueError('Expected one exported kernel')
        def identity(name):
            path = out / name
            return dict(file=str(path.relative_to(operator)), sha256=file_hash(path))
        exports[key] = {**host[0], 'module': identity('kernel.cubin'),
                        'source': identity('kernel.cu'), 'host_abi': identity('host.txt')}

    full_layer = text['layer_types'].index('full_attention')
    for key, slot in [('gates', 'decode/layer0/k3'), ('gatednorm', 'decode/layer0/k6'),
                      ('swiglu', 'decode/layer0/k11'), ('embedding', 'decode/begin/k1'),
                      ('fullprepare', f'decode/layer{full_layer}/k2'), ('capture', 'mtp_capture_m1/body/k0')]:
        export_existing(key, kernels[slot])
    layouts = {}
    for key in ('In', 'Out', 'GateUp', 'Down'):
        b = buffers['L0_' + key + '_P']
        layouts[key] = 'i8' if b['layout'].endswith('mma_i8') else 'f16'
    # Require all layer weights to use the same physical projection contract.
    for layer in range(text['num_hidden_layers']):
        for key in layouts:
            if buffers[f'L{layer}_{key}_P']['layout'] != buffers[f'L0_{key}_P']['layout']:
                raise ValueError('Layer projection layouts differ')
    factories = {
        'norm': lambda: residual_norm(None),
        'qkvz': lambda: w4_small_m(None, 16384, h, TILE_N=128, output_layout='qkvz', weight_layout=layouts['In']),
        'fullproj': lambda: w4_small_m(None, 14336, h, TILE_N=128, weight_layout=layouts['In']),
        'out': lambda: w4_small_m(None, h, 6144, 8, 'float32', TILE_N=128, weight_layout=layouts['Out']),
        'gateup': lambda: w4_small_m(None, 2*f, h, TILE_N=128, weight_layout=layouts['GateUp'],byte_permute=True,vector_words=4),
        'down': lambda: w4_small_m(None, h, f, 8, 'float32', TILE_N=64, weight_layout=layouts['Down'],byte_permute=True,vector_words=4),
        'merge': lambda: split_k_merge(None),
        'ab': lambda: gdn_ab_tensorcore(T.dynamic('M'), BM=16),
        'head': lambda: w4_small_m(None, vocab, h, output_dtype='float32', TILE_N=128),
        'finalnorm': lambda: final_norm(None, 1),
    }
    for key, factory in factories.items():
        compile_kernel(key, factory)

    def emit(program, section, slot, export, pointers, rows, batch=None):
        name = f'{program}/{section}/k{slot}'
        dims = dict(M=rows, rows=rows, batch=rows if batch is None else batch)
        kernels[name] = bind(exports[export], name, pointers, dims)

    def linear(program, rows, layer, batch_slots):
        gdn = text['layer_types'][layer] == 'linear_attention'
        w = lambda suffix: f'L{layer}_{suffix}'
        shared = [
            ('norm', dict(X='Hidden', R='R0', W=w('PreWeight'), Y='Norm', RO='R1')),
            ('qkvz' if gdn else 'fullproj', dict(A='Norm', PP=w('In_P'), S=w('In_S'), Z=w('In_Z'),
              **(dict(QKV='QKV', ZOUT='Zout') if gdn else dict(O='FullX')))),
        ]
        if gdn:
            shared += [('ab', dict(X='Norm', W_ab=w('ABWeight'), Y='AB')),
                       ('gates', dict(A='AB', B='AB', Parameter=w('Al'), DtBias=w('Dt'), G='g', Beta='Beta')),
                       ('gatednorm', dict(X='Y', Z='Zout', W=w('GatedWeight'), Y='MixerIn'))]
        shared += [
            ('out', dict(A='MixerIn', PP=w('Out_P'), S=w('Out_S'), Z=w('Out_Z'), O='Partial')),
            ('merge', dict(P='Partial', O='Mix')),
            ('norm', dict(X='Mix', R='R1', W=w('PostWeight'), Y='Norm', RO='R0')),
            ('gateup', dict(A='Norm', PP=w('GateUp_P'), S=w('GateUp_S'), Z=w('GateUp_Z'), O='GateUp')),
            ('swiglu', dict(X='GateUp', Y='Activated')),
            ('down', dict(A='Activated', PP=w('Down_P'), S=w('Down_S'), Z=w('Down_Z'), O='Partial')),
            ('merge', dict(P='Partial', O='Hidden')),
        ]
        slots = list(range(len(shared))) if batch_slots else ([0,1,2,3,6,7,8,9,10,11,12,13] if gdn else [0,1,5,6,7,8,9,10,11])
        for slot,(key,pointers) in zip(slots,shared):
            emit(program, f'layer{layer}', slot, key, pointers, rows)

    buckets = [2,4,8,16,32,64,128]
    for rows in buckets:
        program = f'batch_m{rows}'
        for layer in range(text['num_hidden_layers']):
            linear(program, rows, layer, True)
        emit(program, 'end', 0, 'head', dict(A='BatchHeadHidden', PP='Head_P', S='Head_S', Z='Head_Z', O='BatchHeadLogits'), rows)

    context = metadata['max_context']
    for rows in (32,64,128):
        program = f'prefill_m{rows}'
        for key,factory in [
            ('prepare', lambda rows=rows: control.prepare(rows)),
            ('advance', lambda rows=rows: control.advance(rows)),
            ('conv', lambda rows=rows: gdn_conv_prep(B=1, tokens=rows, tile_tokens=1)),
            ('gdn', lambda rows=rows: gdn_sequence(rows, in_place=True)),
            ('attention', lambda rows=rows: attention_prefill_int8(1,rows,context,block_m=16,block_n=64,exp_mode='fast')),
        ]:
            compile_kernel(f'{key}_{rows}',factory)
        emit(program,'begin',0,f'prepare_{rows}',dict(Step='Step', Positions='Positions', SeqLength='SeqLength'),rows)
        emit(program,'begin',1,'embedding',dict(P='Embedding_P',S='Embedding_S',Z='Embedding_Z',I='Input',Step='Step',Index='FeatureIndex',Features='Features',Y='Hidden'),rows)
        for layer,kind in enumerate(text['layer_types']):
            w = lambda suffix: f'L{layer}_{suffix}'
            linear(program, rows, layer, False)
            if kind == 'linear_attention':
                emit(program,f'layer{layer}',4,f'conv_{rows}',dict(X='QKV', W=w('ConvWeight'), HI=w('History'),lengths='BatchSegmentLength',positions='Step',Q='Q',K='K',V='V',HO='Ho',positions_out='Po'),rows,batch=1)
                emit(program,f'layer{layer}',5,f'gdn_{rows}',dict(Q='Q',K='K',V='V',G='g',Beta='Beta',State=w('State'),Prefix=w('StatePrefixes'),Out='Y'),rows,batch=1)
            else:
                emit(program,f'layer{layer}',2,'fullprepare',dict(X='FullX',WQ=w('QWeight'),WK=w('KWeight'),Cache='Rotary',Req='Req',Pos='Positions',Pages='Pages',Status='PrepareStatus',MRope='MRopePositions',Q='FullQ',Gate='FullGate',K=w('KPages'),V=w('VPages'),KS=w('KPagesScale'),VS=w('VPagesScale')),rows)
                emit(program,f'layer{layer}',3,f'attention_{rows}',dict(Q='FullQ',K=w('KPages'),V=w('VPages'),Gate='FullGate',Positions='Positions',Lengths='SeqLength',Y='MixerIn',KS=w('KPagesScale'),VS=w('VPagesScale')),rows,batch=1)
        emit(program,'end',0,f'advance_{rows}',dict(Step='Step'),rows)
        head = f'head_m{rows}'
        emit(head,'body',0,'finalnorm',dict(X='Hidden',R='R0',I='BatchLastIndex',W='FinalWeight',Y='LastHidden'),rows,batch=1)
        for slot in (1,2,3):
            k = copy.deepcopy(kernels[f'head_m8/body/k{slot}'])
            k['name'] = f'{head}/body/k{slot}'
            kernels[k['name']] = k
        capture = f'mtp_capture_m{rows}'
        emit(capture,'body',0,'capture',dict(X='Hidden',Residual='R0',Weight='FinalWeight',Step='Step',Out=metadata['mtp']['hidden_ring']),rows)
        metadata['mtp']['capture_plans'].append(dict(tokens=rows,program=capture))
        package['prefill_profiles'].append(dict(tokens=rows,kind='recurrent'))

    def buffer(name,dtype,shape,scope):
        b = dict(name=name,dtype=dtype,shape=shape,layout='contiguous',alignment=256,access='read_write',data=None)
        metadata['buffers'].append(b)
        data['buffer_scopes'][name] = scope
    buffer('BatchHeadHidden','f16',[128,h],'workspace')
    buffer('BatchHeadLogits','f32',[128,vocab],'workspace')
    buffer('BatchSegmentLength','i32',[1],'sequence')
    buffer('BatchLastIndex','i32',[1],'sequence')
    buffers['Partial']['shape'] = [8,128,h]
    for b in metadata['buffers']:
        if b['name'].endswith(('_StatePrefixes','_HistoryPrefixes')):
            data['buffer_scopes'][b['name']] = 'workspace'
    for name in (metadata['logits'],metadata['mtp']['draft_logits']):
        data['buffer_scopes'][name] = 'sequence'
    package['batch_profiles'] = buckets
    package['kernels'] = list(kernels.values())
    package['buffer_contracts'] = [{k:v for k,v in b.items() if k != 'data'} for b in metadata['buffers']]
    digest = commit_package(destination, operator, data, package)
    write_json(report / 'upgrade.json',dict(model=str(destination),operator_package=digest,
        batch_profiles=buckets,weight_bytes=metadata['weight_bytes'],new_kernels=len(exports)))
    print('BATCH PACKAGE READY',destination,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    destination = a.model_output.absolute()
    if destination.exists():
        p.error('Destination already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True,exist_ok=True)
    with atomic_model(destination) as staging:
        upgrade(a.model,staging,a.output)
    report = json.loads((a.output / 'upgrade.json').read_text())
    report['model'] = str(destination)
    write_json(a.output / 'upgrade.json', report)


if __name__ == '__main__':
    main()
