"""Offline PLE lookup using original checkpoint hashes and our E8P rows."""
import bisect
import re

import numpy as np
from tools.model.flash_lookup import RowCache


class PleLookup:
    def __init__(self, checkpoint, *, cache_bytes=32*1024**2):
        self.source = checkpoint
        self.cache = RowCache(checkpoint,capacity_bytes=cache_bytes)
        text = checkpoint.config['text_config']
        self.eos = int(text['eos_token_id'])
        self.vocab = int(text['vocab_size'])
        self.ngram = int(text['ngram_size'])
        self.heads = int(text['heads_per_ngram'])
        names = [n for n in checkpoint.parts if '.ngram_embedding.shard_' in n]
        if not names:raise ValueError('Missing original PLE embedding tensors')
        prefix = names[0].split('.ngram_embedding.')[0]
        self.multipliers = [int(v) for v in checkpoint.tensor(prefix+'.layer_multipliers')]
        self.sizes = [int(v) for v in checkpoint.tensor(prefix+'.ngram_heads_vocab_sizes')]
        self.offsets = [int(v) for v in checkpoint.tensor(prefix+'.ngram_heads_offsets')]
        if len(self.multipliers) != self.ngram or len(self.sizes) != (self.ngram-1)*self.heads or len(self.offsets) != len(self.sizes):
            raise ValueError('Invalid checkpoint n-gram head metadata')
        if any(v < 0 or v*(self.vocab-1) > 2**63-1 for v in self.multipliers) or any(v <= 0 for v in self.sizes):
            raise ValueError('N-gram hash exceeds original signed INT64 bounds')
        indexed = sorted((int(re.search(r'shard_(\d+)\.weight$',n)[1]),n) for n in names)
        if [i for i,_ in indexed] != list(range(len(indexed))):raise ValueError('Missing PLE table shard')
        self.names = [n for _,n in indexed]
        self.starts = [];self.total = 0
        for name in self.names:
            shape = checkpoint.shape(name)
            if len(shape) != 2 or shape[1] != 160:raise ValueError('Invalid native PLE width')
            self.starts.append(self.total);self.total += shape[0]
        if any(o < 0 or o+s > self.total for o,s in zip(self.offsets,self.sizes)):
            raise ValueError('N-gram head outside PLE table')

    def row_ids(self, tokens, history):
        if any(type(t) is not int or not 0 <= t < self.vocab for t in tokens):
            raise ValueError('Token outside original PLE vocabulary')
        if len(history) >= self.ngram or any(type(t) is not int or not 0 <= t < self.vocab or t == self.eos for t in history):
            raise ValueError('Invalid request-owned PLE history')
        next_history = list(history)
        rows = []
        for token in tokens:
            mixed = token*self.multipliers[0]
            for shift in range(1,self.ngram):
                previous = next_history[-shift] if len(next_history) >= shift else self.eos
                mixed ^= previous*self.multipliers[shift]
                first = (shift-1)*self.heads
                rows.extend(mixed%self.sizes[i]+self.offsets[i] for i in range(first,first+self.heads))
            next_history = [] if token == self.eos else (next_history+[token])[-(self.ngram-1):]
        return rows,next_history

    def prepare(self, tokens, history):
        rows,next_history = self.row_ids(tokens,history)
        # Deduplicate only immutable table reads, preserving requested order.
        values = {}
        requests={}
        for row in sorted(set(rows)):
            shard = bisect.bisect_right(self.starts,row)-1
            requests[row]=(self.names[shard],row-self.starts[shard],1)
        values={row:self.cache.read(request[0],request[1]) for row,request in requests.items()}
        features = np.stack([values[row] for row in rows]).reshape(len(tokens),-1)
        return features,next_history
