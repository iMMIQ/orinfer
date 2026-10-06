"""Offline PLE lookup using original checkpoint hashes and our E8P rows."""
import bisect
import re

import numpy as np
from tools.model.flash_lookup import RowCache


class PleLookup:
    def __init__(self, checkpoint, *, cache_bytes=32*1024**2, row_dtype=np.float32):
        self.source = checkpoint
        self.cache = RowCache(checkpoint,capacity_bytes=cache_bytes,dtype=row_dtype)
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
        if len(tokens)>=256:
            # Products stay in signed INT64 by the checkpoint bounds checked
            # above. Reset boundaries suppress every pre-EOS history token.
            width=self.ngram-1
            context=np.asarray([self.eos]*(width-len(history))+list(history)+list(tokens),np.int64)
            positions=np.arange(width,len(context))
            resets=np.maximum.accumulate(np.where(context==self.eos,np.arange(len(context)),-1))
            mixed=context[positions]*self.multipliers[0]
            rows=np.empty((len(tokens),width*self.heads),np.int64)
            for shift in range(1,self.ngram):
                previous=np.where(positions-shift>resets[positions-1],context[positions-shift],self.eos)
                mixed=np.bitwise_xor(mixed,previous*self.multipliers[shift])
                first=(shift-1)*self.heads
                rows[:,first:first+self.heads]=mixed[:,None]%np.asarray(self.sizes[first:first+self.heads])+np.asarray(self.offsets[first:first+self.heads])
            next_history=context[max(int(resets[-1])+1,len(context)-width):].tolist()
            return rows.reshape(-1).tolist(),next_history
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
        unique,inverse=np.unique(rows,return_inverse=True)
        values=[]
        for row in unique:
            row=int(row)
            shard = bisect.bisect_right(self.starts,row)-1
            values.append(self.cache.read(self.names[shard],row-self.starts[shard]))
        features = np.stack(values)[inverse].reshape(len(tokens),-1)
        return features,next_history
