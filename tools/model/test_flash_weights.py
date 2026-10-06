import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from tools.model.flash_weights import coverage, check_piece, payload_hash, publish
from tools.model.flash_checkpoint import Checkpoint
from tools.quantization.e8p import basis
from tools.quantization.embedding_vq import pack, decode
from tools.quantization.flash_next import digest


class FakeSource:
    def __init__(self, tensors):
        self.tensors = tensors
        self.weight_map = {name:'original.safetensors' for name in tensors}

    def tensor(self, name):
        return 'original.safetensors',0,self.tensors[name]


class PublishedFlashWeightsTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root);converted = root/'converted';converted.mkdir()
        dense = 'lm_head.weight'
        gamma = 'model.language_model.hyper_connection_mixer.hc_norm.weight'
        embedding = 'model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight'
        expert = 'model.language_model.layers.0.mlp.experts.gate_up_proj'
        source = FakeSource({dense:{'shape':[5,4],'dtype':'BF16'},gamma:{'shape':[3],'dtype':'BF16'},
                             embedding:{'shape':[4,160],'dtype':'BF16'},expert:{'shape':[2,2,128],'dtype':'BF16'}})
        records = []
        for i,(first,count) in enumerate(((0,2),(2,3))):
            weights = np.arange(first*4,(first+count)*4,dtype=np.int8).reshape(count,4)
            scales = np.full(count,.25,np.float16)
            records.append(self.write(converted,source,dense,first,count,'int8-row',{'weight':weights,'scale':scales}))
        raw = np.array([0x3f80,0xbf80,0x4000],np.uint16).tobytes()
        records.append(self.write(converted,source,gamma,0,3,'original',raw))
        codes = np.arange(80,dtype=np.uint16).reshape(4,20)
        scales = np.full(4,.015625,np.float16)
        signs = np.ones(160,np.int8)
        records.append(self.write(converted,source,embedding,0,4,'e8p-embedding',{'rows':pack(codes,scales),'table':basis(),'signs':signs}))
        records.append(self.write(converted,source,expert,0,2,'e8p-expert',
                                  {'indices':np.arange(64,dtype=np.uint16).reshape(2,1,2,16),
                                   'scales':np.full((2,2),.01,np.float16),'table':basis(),'signs':np.ones(128,np.int8)}))
        records.sort(key=lambda r:(r['tensor'],r['first']))
        config = root/'config.json';config.write_text(json.dumps({'model_type':'qwen4_exp','text_config':{'num_hidden_layers':1}}))
        frontend = root/'frontend';frontend.mkdir()
        for name in ('tokenizer.json','tokenizer_config.json'):(frontend/name).write_text('{}')
        contract = {'layers':[0,1],'basis':'integer-e8p-spread29-v1','rotation':'signed-block128',
                    'source':'test/original','revision':'0'*40,'seed':20261002}
        return converted,source,records,config,frontend,contract

    def write(self, directory, source, name, first, count, kind, data):
        shape = source.tensors[name]['shape']
        path = directory/f'{len(list(directory.iterdir()))}.safetensors'
        source_raw = data if isinstance(data,bytes) else bytes(count*int(np.prod(shape[1:]))*2)
        source_hash = hashlib.sha256(source_raw).hexdigest()
        meta = {'source_tensor':name,'source_range_sha256':source_hash}
        record = {'tensor':name,'kind':kind,'first':first,'count':count,'filename':path.name,
                  'source_range_sha256':source_hash}
        if kind == 'e8p-expert':
            record.update(group='experts',shape=[count,*shape[1:]],first_expert=first)
            meta.update(first_expert=str(first),logical_shape=json.dumps(record['shape']))
        else:
            record.update(group='aux',source_shape=shape,dtype='BF16',first_row=first,rows=count)
            meta.update(first_row=str(first),rows=str(count))
        if kind == 'original':
            header = json.dumps({'__metadata__':meta,name:{'dtype':'BF16','shape':shape,'data_offsets':[0,len(data)]}}).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header+data)
        else:save_file(data,str(path),metadata=meta)
        record['sha256'] = digest(path)
        return record

    def test_detects_holes_overlap_and_wrong_source_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            converted,source,records,*_ = self.fixture(d)
            self.assertTrue(coverage(source,records)['complete'])
            self.assertFalse(coverage(source,records[1:])['complete'])
            with self.assertRaises(ValueError):coverage(source,records+[records[0]])
            wrong = copy.deepcopy(records);wrong[0]['count'] = 100
            with self.assertRaises(ValueError):coverage(source,wrong)
            for r in records:check_piece(converted/r['filename'],r,source)
            path = converted/records[0]['filename']
            with path.open('r+b') as f:
                f.seek(-1,2);value = f.read(1);f.seek(-1,2);f.write(bytes([value[0]^1]))
            with self.assertRaises(ValueError):check_piece(path,records[0],source)

    def test_standard_index_lossless_payload_consumption_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            converted,source,records,config,frontend,contract = self.fixture(d)
            before = {r['filename']:payload_hash(converted/r['filename']) for r in records}
            output = Path(d)/'published'
            with patch('tools.model.flash_weights.source_and_records',return_value=(source,contract,records)):
                state = publish(converted,output,'unused-index',config,frontend)
                self.assertTrue(state['complete'])
                self.assertFalse(state['full_model_quality_verified'])
                self.assertFalse(any(converted.glob('*.safetensors')))
                for name,r in state['shards'].items():self.assertEqual(payload_hash(output/name),before[r['source_filename']])
                repeated = publish(converted,output,'unused-index',config,frontend)
                self.assertEqual(state,repeated)
            model = Checkpoint(output)
            np.testing.assert_array_equal(model.rows('lm_head.weight',1,3),np.arange(4,16).reshape(3,4)*.25)
            np.testing.assert_array_equal(model.tensor('model.language_model.hyper_connection_mixer.hc_norm.weight'),[1,-1,2])
            embedding = next(n for n in source.tensors if 'ngram_embedding' in n)
            expected = decode(pack(np.arange(80,dtype=np.uint16).reshape(4,20),np.full(4,.015625,np.float16)),basis(),np.ones(160,np.int8))
            np.testing.assert_array_equal(model.rows(embedding,1,2),expected[1:3])
            expert = next(n for n in source.tensors if '.experts.' in n)
            pieces = list(model.expert_parts(expert))
            self.assertEqual(len(pieces),1);self.assertEqual(pieces[0][0],0)
            np.testing.assert_array_equal(pieces[0][1]['indices'],np.arange(64,dtype=np.uint16).reshape(2,1,2,16))
            index = json.loads((output/'model.safetensors.index.json').read_text())
            key = next(iter(index['weight_map']));index['weight_map'][key] = '../escape.safetensors'
            (output/'model.safetensors.index.json').write_text(json.dumps(index))
            with self.assertRaises(ValueError):Checkpoint(output)

    def test_incomplete_conversion_does_not_consume_any_weight(self):
        with tempfile.TemporaryDirectory() as d:
            converted,source,records,config,frontend,contract = self.fixture(d)
            before = list(converted.iterdir())
            with patch('tools.model.flash_weights.source_and_records',return_value=(source,contract,records[1:])):
                with self.assertRaises(ValueError):publish(converted,Path(d)/'published','unused',config,frontend)
            self.assertEqual(list(converted.iterdir()),before)


if __name__ == '__main__':unittest.main()
