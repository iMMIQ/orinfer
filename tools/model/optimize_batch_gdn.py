"""Add private-arena batched M1 GDN mixers without changing resident weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from tools.model.prepare import file_hash, write_json
from tools.model.upgrade_batching import bind
from tools.operators.abi import parse_host


def upgrade(model, destination, report):
    from kernels.model.gdn_batch import batch_gdn_conv, batch_gdn_recurrent
    from tools.operators.common import configure, export_kernel
    configure()
    model = model.resolve(strict=True)
    data = json.loads((model/'cache/model.json').read_text())
    metadata = data['metadata']
    text_config = json.loads((model/'config.json').read_text())
    text = text_config.get('text_config',text_config)
    dims = [text[k] for k in ['linear_num_key_heads','linear_num_value_heads',
                              'linear_key_head_dim','linear_value_head_dim','linear_conv_kernel_dim']]
    if dims != [16,48,128,128,4]:
        raise ValueError('Batched GDN factories require the 27B mixer dimensions')
    root = Path(os.environ.get('ORIN_OPERATOR_CACHE',
        str(Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'orin-llm/operators')))
    origin = root/data['operator_package']
    if not origin.exists():
        origin = model/'cache/operators'/data['operator_package']
    if file_hash(origin/'package.json') != data['operator_package']:
        raise ValueError('Source operator package digest mismatch')
    package = json.loads((origin/'package.json').read_text())
    if not package.get('batch_profiles') or package.get('batch_gdn'):
        raise ValueError('Requires a batch package without batched GDN')
    buffers = {b['name']:b for b in metadata['buffers']}
    for layer,kind in enumerate(text['layer_types']):
        if kind == 'linear_attention':
            for suffix,dtype,shape in [('State','f32',[1,48,128,128]),('History','f16',[1,3,10240])]:
                b = buffers[f'L{layer}_{suffix}']
                if b['dtype'] != dtype or b['shape'] != shape or data['buffer_scopes'][b['name']] != 'sequence':
                    raise ValueError('Unsupported private mixer layout')
    destination.mkdir(parents=True)
    for path in model.iterdir():
        if path.is_file():
            shutil.copyfile(path,destination/path.name)
    cache = destination/'cache';cache.mkdir()
    shutil.copytree(model/'cache/weights',cache/'weights',copy_function=os.link)
    operator = cache/'operators'/'.building'
    shutil.copytree(origin,operator,copy_function=os.link)
    kernels = list(package['kernels'])
    for rows in package['batch_profiles']:
        for slot,factory in enumerate([batch_gdn_conv,batch_gdn_recurrent]):
            out = operator/'batch-gdn-aot'/f'm{rows}-k{slot}'
            print('compile',out.name,flush=True)
            export_kernel(factory(rows),out)
            abi = parse_host((out/'host.txt').read_text())
            if len(abi) != 1:
                raise ValueError('Expected a single kernel export')
            def identity(filename):
                p = out/filename
                return dict(file=str(p.relative_to(operator)),sha256=file_hash(p))
            export = dict(**abi[0],module=identity('kernel.cubin'),source=identity('kernel.cu'),host_abi=identity('host.txt'))
            for layer,kind in enumerate(text['layer_types']):
                if kind != 'linear_attention':
                    continue
                names = dict(Pointers='BatchGdnPointers',X='QKV',W=f'L{layer}_ConvWeight',
                             Q='Q',K='K',V='V',G='g',Beta='Beta',Out='Y')
                kernels.append(bind(export,f'batch_gdn_m{rows}/layer{layer}/k{slot}',names,{}))
    spec = dict(name='BatchGdnPointers',dtype='u64',shape=[text['num_hidden_layers'],128,3],
                layout='contiguous',alignment=256,access='read_write',data=None)
    metadata['buffers'].append(spec)
    data['buffer_scopes'][spec['name']] = 'workspace'
    package['batch_gdn'] = True
    package['kernels'] = kernels
    package['buffer_contracts'] = [{k:v for k,v in b.items() if k!='data'} for b in metadata['buffers']]
    raw = (json.dumps(package,ensure_ascii=False,indent=2)+'\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    tmp = operator/'package.new.json';tmp.write_bytes(raw);tmp.replace(operator/'package.json')
    operator.rename(operator.parent/digest)
    data['operator_package'] = digest
    write_json(cache/'model.json',data)
    write_json(report/'upgrade.json',dict(model=str(destination),operator_package=digest,
        batch_profiles=package['batch_profiles'],weight_bytes=metadata['weight_bytes']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    destination = a.model_output.absolute()
    if destination.exists():
        p.error('Destination already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging = destination.with_name('.'+destination.name+'.building-'+uuid.uuid4().hex)
    a.output.mkdir(parents=True,exist_ok=True)
    try:
        upgrade(a.model,staging,a.output)
        cli = Path(__file__).resolve().parents[2]/'target/release/orin-llm'
        subprocess.run([str(cli),'plan-model',str(staging)],check=True,stdout=subprocess.DEVNULL)
        staging.rename(destination)
        report = json.loads((a.output/'upgrade.json').read_text())
        report['model'] = str(destination)
        write_json(a.output/'upgrade.json',report)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == '__main__':
    main()
