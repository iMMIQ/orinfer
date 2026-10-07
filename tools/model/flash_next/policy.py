"""CPU policy for draft-only vocabulary pruning and bounded adaptive depths."""
from collections import Counter


def vocabulary(frequencies, required, size, vocab):
    if type(size) is not int or not 256<=size<=vocab or size%64:
        raise ValueError('Draft vocabulary size must be a multiple of 64 in [256,V]')
    required=set(required)
    if any(type(t) is not int or not 0<=t<vocab for t in required|set(frequencies)):
        raise ValueError('Invalid draft vocabulary token')
    if len(required)>size:raise ValueError('Required tokens exceed draft vocabulary')
    selected=set(required)
    ranked=sorted(frequencies,key=lambda t:(-frequencies[t],t))
    for token in ranked:
        if len(selected)==size:break
        selected.add(token)
    for token in range(vocab):
        if len(selected)==size:break
        selected.add(token)
    # Ascending original IDs preserve deterministic greedy tie-breaking.
    return sorted(selected)


def code_vocabulary(tokenizer, root, size):
    """Use project source and authored prose; never measured continuations."""
    counts=Counter()
    for directory in ('crates','kernels'):
        for path in sorted((root/directory).rglob('*')):
            if path.suffix not in ('.rs','.py') or not path.is_file():continue
            counts.update(tokenizer.encode(path.read_text()[:32768],add_special_tokens=False))
    counts.update(tokenizer.encode(
        'Let me think through this carefully. First check the requirements, edge cases, '
        'correctness, complexity, synchronization, tests and implementation. '
        '需要先分析需求、边界条件、异常处理、线程安全和测试，再给出完整实现。',add_special_tokens=False))
    required=set(tokenizer.all_special_ids)
    required.update(tokenizer.encode(''.join(chr(i) for i in range(32,127))+'\n\t',add_special_tokens=False))
    return vocabulary(counts,required,size,len(tokenizer))


class AdaptiveDepth:
    """Choose complete rounds by predicted committed tokens per measured ms.

    Survival and costs use independent EMAs. Occasional neighboring-tier probes
    learn unobserved tails; hysteresis bounds switching. Instances are request
    owned and reset on replay. No policy decision affects target acceptance.
    """
    def __init__(self, initial=3, candidates=(1,3,7), *, interval=8, alpha=.2):
        if initial not in candidates or any(type(k) is not int or not 1<=k<=7 for k in candidates):
            raise ValueError('Invalid adaptive depth tiers')
        if len(set(candidates))!=len(candidates) or interval<1 or not 0<alpha<=1:
            raise ValueError('Invalid adaptive policy')
        self.candidates=tuple(sorted(candidates));self.current=initial
        self.interval,self.alpha=interval,alpha
        self.survival=[.8**j for j in range(1,8)];self.costs={};self.rounds=0
        self.visits=Counter();self.probe=None

    def observe(self, depth, proposed, accepted, elapsed_ms):
        if depth not in self.candidates or not 0<=accepted<=proposed<=depth or elapsed_ms<=0:
            raise ValueError('Invalid adaptive round observation')
        # A clipped tail is not representative of a complete tier.
        if proposed!=depth:return
        a=self.alpha
        for j in range(proposed):self.survival[j]=(1-a)*self.survival[j]+a*(accepted>j)
        self.costs[depth]=elapsed_ms if depth not in self.costs else (1-a)*self.costs[depth]+a*elapsed_ms
        self.visits[depth]+=1;self.rounds+=1

    def choose(self):
        if self.probe is not None:
            self.probe=None
            return self.current
        if self.rounds and self.rounds%self.interval==0:
            unknown=[k for k in self.candidates if not self.visits[k]]
            if unknown:
                self.probe=min(unknown,key=lambda k:abs(k-self.current));return self.probe
            scores={k:(1+sum(self.survival[:k]))/self.costs[k] for k in self.candidates}
            best=max(self.candidates,key=lambda k:scores[k])
            if scores[best]>scores[self.current]*1.08:self.current=best
            if self.rounds%(4*self.interval)==0:
                alternatives=[k for k in self.candidates if k!=self.current]
                if alternatives:
                    self.probe=min(alternatives,key=lambda k:self.visits[k]);return self.probe
        return self.current
