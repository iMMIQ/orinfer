import unittest

import numpy as np

from tools.model.flash_ple import PleLookup


class Table:
    def __init__(self, metadata):
        prefix = 'model.language_model.layers.1.ple.ple_embedding'
        self.prefix = prefix
        self.config = {'text_config':{'eos_token_id':248044,'vocab_size':248320,'ngram_size':3,'heads_per_ngram':8}}
        self.data = {prefix+'.layer_multipliers':metadata['multipliers'],
                     prefix+'.ngram_heads_offsets':np.cumsum([0,*metadata['sizes'][:-1]]).tolist(),
                     prefix+'.ngram_heads_vocab_sizes':metadata['sizes']}
        self.parts = {prefix+'.ngram_embedding.shard_0.weight':[],prefix+'.ngram_embedding.shard_1.weight':[]}
        self.total = metadata['table_rows'];self.reads = []

    def tensor(self, name):return self.data[name]

    def shape(self, name):
        return (self.total//2,160)

    def rows(self, name, first, count):
        shard = 0 if name.endswith('shard_0.weight') else 1
        row = shard*(self.total//2)+first
        self.reads.append(row)
        return np.full((count,160),row,np.float32)


class FlashPleTests(unittest.TestCase):
    def metadata(self):
        # Frozen independent SGLang reference, not derived by PleLookup.
        return {'multipliers':[23703573157769,20109073645365,8052911324071],
                'sizes':[20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,
                         20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171],
                'table_rows':320001536}

    def test_chunking_eos_and_private_history(self):
        lookup = PleLookup(Table(self.metadata()))
        tokens = [100,101,248044,102,248319,0]
        all_rows,history = lookup.row_ids(tokens,[])
        prefix,first_history = lookup.row_ids(tokens[:3],[])
        tail,second_history = lookup.row_ids(tokens[3:],first_history)
        self.assertEqual(all_rows,prefix+tail);self.assertEqual(history,second_history)
        self.assertEqual(first_history,[])
        original = [100,101]
        lookup.row_ids([102],original);self.assertEqual(original,[100,101])
        with self.assertRaises(ValueError):lookup.row_ids([248320],original)
        self.assertEqual(original,[100,101])

    def test_original_signed64_hash_formula_and_row_lookup(self):
        table = Table(self.metadata());lookup = PleLookup(table)
        row_ids,history = lookup.row_ids([100,101],[])
        multipliers = np.asarray(self.metadata()['multipliers'],np.int64)
        context = np.array([[100,248044,248044],[101,100,248044]],np.int64)
        hashes = np.bitwise_xor.accumulate(context*multipliers[None,:],axis=1)[:,1:]
        sizes = np.asarray(self.metadata()['sizes'],np.int64).reshape(2,8)
        offsets = np.asarray(table.tensor(table.prefix+'.ngram_heads_offsets'),np.int64).reshape(2,8)
        expected = (hashes[:,:,None]%sizes[None,:,:]+offsets[None,:,:]).reshape(-1).tolist()
        self.assertEqual(row_ids,expected)
        self.assertEqual(row_ids[:16],[5727835,21884476,43702108,66434789,84094509,104112171,134886528,157314943,
                                     162363667,189466439,200618315,232121522,258547276,266199001,289022207,305180972])
        features,new_history = lookup.prepare([100,101],[])
        np.testing.assert_array_equal(features,np.repeat(np.array(row_ids,np.float32).reshape(2,16),160,axis=1))
        self.assertEqual(new_history,history)
        self.assertEqual(len(table.reads),len(set(row_ids)))

    def test_immutable_cache_and_bounded_eviction(self):
        table=Table(self.metadata());lookup=PleLookup(table)
        expected,history=lookup.prepare([100,101],[])
        reads=len(table.reads)
        actual,_=lookup.prepare([100,101],[])
        self.assertEqual(len(table.reads),reads)
        np.testing.assert_array_equal(expected,actual)
        actual.fill(0)
        again,_=lookup.prepare([100,101],[])
        np.testing.assert_array_equal(expected,again)
        tiny=PleLookup(table,cache_bytes=1200)
        small,_=tiny.prepare([100,101],[])
        np.testing.assert_array_equal(expected,small)
        self.assertLessEqual(tiny.cache.bytes,1200)
        self.assertLessEqual(len(tiny.cache.rows),1)
        disabled=PleLookup(table,cache_bytes=0)
        np.testing.assert_array_equal(expected,disabled.prepare([100,101],[])[0])
        self.assertEqual(disabled.cache.bytes,0)

    def test_vectorized_hashes_preserve_reset_and_chunk_boundaries(self):
        lookup=PleLookup(Table(self.metadata()))
        rng=np.random.default_rng(20261002)
        tokens=rng.integers(0,lookup.vocab,size=1025).tolist()
        for position in (0,1,254,255,256,257,511,512,1024):tokens[position]=lookup.eos
        for history in ([],[100],[100,101]):
            original=list(history)
            rows,after=lookup.row_ids(tokens,history)
            expected=[];serial=list(history)
            for token in tokens:
                part,serial=lookup.row_ids([token],serial);expected.extend(part)
            self.assertEqual(rows,expected);self.assertEqual(after,serial)
            first,middle=lookup.row_ids(tokens[:513],history)
            tail,end=lookup.row_ids(tokens[513:],middle)
            self.assertEqual(rows,first+tail);self.assertEqual(after,end)
            self.assertEqual(history,original)

    def test_half_cache_preserves_operand_rounding_and_immutability(self):
        from tools.model.flash_lookup import RowCache
        rng=np.random.default_rng(20261002)
        weights=rng.normal(size=(8,160)).astype(np.float32)
        class Weights:
            def rows(self,name,first,count):return weights[first:first+count]
        source=Weights();full=RowCache(source,capacity_bytes=8192)
        half=RowCache(source,capacity_bytes=8192,dtype=np.float16)
        for row in range(len(weights)):
            expected=full.read('test',row).astype(np.float16)
            actual=half.read('test',row)
            np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
            self.assertFalse(actual.flags.writeable)
        self.assertLessEqual(half.bytes,half.capacity)
        with self.assertRaises(ValueError):RowCache(source,capacity_bytes=8192,dtype=np.int8)


if __name__ == '__main__':unittest.main()
