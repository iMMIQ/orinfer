import unittest
import json
from pathlib import Path
import struct
import tempfile
import threading
from unittest.mock import patch

import numpy as np

from tools.quantization.flash_next_aux import convert, int8_rows, policy, prefetch
from tools.quantization.e8p import basis
from tools.quantization.flash_next import equivalent
from tools.model.safetensors_source import Source
from safetensors import safe_open


class AuxiliaryQuantizationTests(unittest.TestCase):
    def test_prefetch_preserves_order_and_bounds_unconsumed_reads(self):
        class Reader:
            def __init__(self):self.seen=[];self.second=threading.Event();self.lock=threading.Lock()
            def tensor(self,name):return name,0,{}
            def rows(self,name,first,count):
                with self.lock:self.seen.append(first)
                if first == 1:self.second.set()
                if first == 0 and not self.second.wait(2):raise AssertionError('Second read did not overlap first')
                return bytes([first])
        queue=[{'tensor':'rows','kind':'int8-row','first_row':i,'rows':1} for i in range(5)]
        source=Reader();reads=prefetch(source,queue)
        task,(raw,seconds)=next(reads)
        self.assertEqual((task['first_row'],raw),(0,b'\0'))
        self.assertEqual(sorted(source.seen),[0,1])
        reads.close()
        self.assertEqual(sorted(source.seen),[0,1])
        self.assertEqual([raw for _,(raw,_) in prefetch(Reader(),queue)],list(map(lambda i:bytes([i]),range(5))))
        for workers in (0,3,True):
            with self.assertRaises(ValueError):next(prefetch(source,queue,workers=workers))
        class LargeCritical:
            def tensor(self,name):return name,0,{'data_offsets':[0,64*1024**2+1]}
            def read(self,*args):raise AssertionError('Oversized critical payload was read')
        with self.assertRaises(ValueError):
            next(prefetch(LargeCritical(),[{'tensor':'large','kind':'original'}]))

    def test_prefetched_mixed_payloads_match_serial_and_resume_without_overfetch(self):
        class Encoder:
            table=basis()
            def fit_arrays(self,raw,signs,*,rotation):
                if rotation != 'full':raise AssertionError('Wrong embedding rotation')
                return np.zeros((len(raw),raw.shape[1]//8),np.uint16),np.ones(len(raw),np.float16)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);names=[
                'model.language_model.hyper_connection_mixer.hc_norm.weight',
                'model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
                'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight']
            shapes=[(3,),(2,128),(2,160)];header={};payload=bytearray()
            for name,shape in zip(names,shapes):
                value=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)/64-.5
                raw=(value.view(np.uint32)>>16).astype('<u2').tobytes()
                first=len(payload);payload.extend(raw)
                header[name]={'dtype':'BF16','shape':list(shape),'data_offsets':[first,len(payload)]}
            encoded=json.dumps(header).encode()
            (root/'source.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
            index=root/'index.json';index.write_text(json.dumps({'weight_map':{name:'source.safetensors' for name in names}}))
            source=Source(index,directory=root)
            for name in names:source.tensor(name)
            parallel=root/'parallel';serial=root/'serial';parallel.mkdir();serial.mkdir()
            # Limit/filter before submission: even the next missing payload
            # must not be fetched by this partial run.
            with patch.object(source,'read',wraps=source.read) as reads:
                first=convert(source,parallel,Encoder(),source_contract='test',seed=20261002,
                              kind_filter='int8-row',max_chunks=1)
                self.assertEqual(reads.call_count,1)
                self.assertEqual(reads.call_args.args[2],2*128*2)
            saved=next(iter(first['shards'].values()))['sha256']
            complete=convert(source,parallel,Encoder(),source_contract='test',seed=20261002)
            self.assertTrue(complete['complete'])
            self.assertIn(saved,[record['sha256'] for record in complete['shards'].values()])
            with patch.object(source,'read',side_effect=AssertionError('Resume reread completed payload')):
                self.assertEqual(convert(source,parallel,Encoder(),source_contract='test',seed=20261002),complete)
            with patch('tools.quantization.flash_next_aux.prefetch',
                       side_effect=lambda source,queue:prefetch(source,queue,workers=1)):
                expected=convert(source,serial,Encoder(),source_contract='test',seed=20261002)
            self.assertEqual(set(expected['shards']),set(complete['shards']))
            for name in expected['shards']:
                # safetensors metadata key order can vary between saves;
                # require identical parsed headers and exact payload bytes.
                self.assertTrue(equivalent(serial/name,parallel/name))

    def test_dense_rows_rne_zero_and_small_error(self):
        x = np.array([[0,0,0],[-1,0,1],[-.001,.002,.003]],np.float32)
        q,s = int8_rows(x)
        self.assertEqual(q.dtype,np.int8)
        np.testing.assert_array_equal(q[0],0)
        np.testing.assert_allclose(q*s.astype(np.float32)[:,None],x,atol=.004)
        with self.assertRaises(ValueError):int8_rows(np.full((1,128),np.nan))

    def test_critical_weights_keep_original_precision(self):
        info = {'shape':[512,2560]}
        self.assertEqual(policy('model.language_model.layers.0.mlp.gate.weight',info),'original')
        self.assertEqual(policy('model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight',info),'original')
        self.assertEqual(policy('model.language_model.layers.0.linear_attn.in_proj_qkv.weight',info),'int8-row')
        self.assertEqual(policy('model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight',info),'e8p-embedding')

    def test_original_bf16_payload_and_resume_without_numpy_bf16_cast(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            name = 'model.language_model.hyper_connection_mixer.hc_norm.weight'
            raw = np.array([0x3f80,0xbf80,0x4000],np.uint16).tobytes()
            header = json.dumps({name:{'dtype':'BF16','shape':[3],'data_offsets':[0,len(raw)]}}).encode()
            (root/'source.safetensors').write_bytes(struct.pack('<Q',len(header))+header+raw)
            (root/'index.json').write_text(json.dumps({'weight_map':{name:'source.safetensors'}}))
            output = root/'output';output.mkdir()
            source = Source(root/'index.json',directory=root)
            first = convert(source,output,None,source_contract='test',seed=20261002)
            self.assertTrue(first['complete'])
            record = next(iter(first['shards'].values()))
            path = output/record['filename']
            with safe_open(path,framework='np') as f:
                self.assertEqual(f.get_slice(name).get_dtype(),'BF16')
            with path.open('rb') as f:
                size, = struct.unpack('<Q',f.read(8));f.seek(8+size)
                self.assertEqual(f.read(),raw)
            self.assertEqual(convert(source,output,None,source_contract='test',seed=20261002),first)


if __name__ == '__main__':unittest.main()
