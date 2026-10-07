"""Native Flash Next greedy MTP with complete target-prefix commit and restore.

All GPU math is TileLang. The draft shares the target embedding cache and
output weights, keeps its own INT8 KV/index, and never controls target tokens.
This is the offline Flash recipe; online Rust serving remains separate.
"""
import torch
from contextlib import nullcontext
import time

from tools.model.flash_next.chunks import chunks
from tools.model.flash_next.speculation import DEFAULT_DRAFTS,greedy_commit,clip_outputs,verification_size
from tools.model.flash_next.validation import validate_snapshot


class Session:
    def __init__(self, target, draft):
        if target.is_mtp or not draft.is_mtp or target.capacity!=draft.capacity:
            raise ValueError('Expected matching target and MTP runtimes')
        self.target,self.draft=target,draft
        self.pending=None;self.draft_token=None;self.hidden=None
        self.statistics={}

    def prefill(self, tokens, chunk=4096, *, draft_chunk=512):
        if not tokens or len(tokens)>=self.target.capacity:raise ValueError('Prompt must leave output context')
        if type(draft_chunk) is not int or not 1<=draft_chunk<=4096:raise ValueError('Invalid draft prefill chunk')
        self.target.reset();self.draft.reset()
        cursor=0
        for batch in chunks(tokens,chunk):
            end=cursor+len(batch)
            token=self.target.execute(batch,output='token' if end==len(tokens) else 'none')
            shifted=list(tokens[cursor+1:end+1])
            if end==len(tokens):shifted.append(token)
            # The target benefits from larger expert batches. Stream its saved
            # HC rows through the draft without duplicating a large arena.
            condition=self.target.last_plan['residual']
            draft_cursor=0
            for part in chunks(shifted,min(chunk,draft_chunk)):
                draft_end=draft_cursor+len(part)
                prediction=self.draft.execute(part,hidden=condition[draft_cursor:draft_end],
                    output='token' if end==len(tokens) and draft_end==len(shifted) else 'none')
                draft_cursor=draft_end
            cursor=end
        self.pending=token;self.draft_token=prediction
        self.hidden=self.draft.last_plan['residual'][-1:].clone()
        self.statistics={'rounds':0,'proposed':0,'accepted':0,'fallback_steps':0}
        return token

    def warm(self, token, depths=(3,)):
        """Compile/capture all verification tails, commits and draft refreshes."""
        if isinstance(depths,int):depths=(depths,)
        maximum=max(depths)
        widths=sorted({verification_size(depth,n,self.target.capacity)
                       for depth in depths for n in range(1,depth+2)})
        for width in widths:
            if width>self.target.capacity:continue
            for accepted in range(1,width+1):
                self.target.reset()
                self.target.execute([token]*width,output='verify' if width>1 else 'token')
                if width>1:self.target.commit(accepted)
        condition=self.target.last_plan['residual']
        for width in range(1,min(maximum+1,len(condition))+1):
            self.draft.reset();self.draft.execute([token]*width,hidden=condition[:width],output='token')
        self.target.reset();self.draft.reset();self.pending=self.draft_token=self.hidden=None

    def generate(self, budget, eos, *, drafts=DEFAULT_DRAFTS, override=None, policy=None, trace=None):
        """Emit a target token, then verified rounds. override is validation-only."""
        verification_size(drafts,budget,max(1,self.target.capacity-self.target.position))
        if budget>self.target.capacity-self.target.position+1:raise ValueError('Generation exceeds context capacity')
        if self.pending is None or self.target.position!=self.draft.position:raise ValueError('Prefill or restore a complete MTP session first')
        generated,reason=clip_outputs([self.pending],eos,budget)
        if reason is not None:return generated,reason
        while len(generated)<budget:
            depth=policy.choose() if policy else drafts
            size=verification_size(depth,budget-len(generated),self.target.capacity-self.target.position)
            begin=time.perf_counter() if policy else None
            phase=trace.phase if trace else lambda name:nullcontext()
            if size==1:
                self.pending=self.target.execute([self.pending],output='token')
                self.draft_token=self.draft.execute([self.pending],hidden=self.target.last_plan['residual'][-1:],output='token')
                self.hidden=self.draft.last_plan['residual'][-1:].clone()
                self.statistics['fallback_steps']+=1
                values,reason=clip_outputs([self.pending],eos,budget-len(generated))
                generated.extend(values)
                if reason is not None:break
                continue
            start=self.target.position
            pending_state=self.draft.states['48:pending'].clone()
            proposals=[self.draft_token]
            if override is not None:proposals[0]=int(override(self.statistics['rounds'],0,proposals[0]))
            hidden=self.hidden
            with phase('draft'):
                for index in range(1,size-1):
                    value=self.draft.execute([proposals[-1]],hidden=hidden,output='token')
                    hidden=self.draft.last_plan['residual'][-1:]
                    if override is not None:value=int(override(self.statistics['rounds'],index,value))
                    proposals.append(value)
            with phase('verify'):predictions=self.target.execute([self.pending]+proposals,output='verify')
            committed=greedy_commit(proposals,predictions)
            values,reason=clip_outputs(committed,eos,budget-len(generated))
            with phase('commit'):self.target.commit(len(values))
            # The first draft slot already used the true target condition.
            # Discard only future proposal slots and recompute them with true HC.
            self.draft.position=start;self.draft.position_gpu.fill_(start)
            self.draft.states['48:pending'].copy_(pending_state)
            with phase('refresh'):
                self.draft_token=self.draft.execute(values,hidden=self.target.last_plan['residual'][:len(values)],output='token')
            self.hidden=self.draft.last_plan['residual'][-1:].clone()
            self.pending=values[-1]
            self.statistics['rounds']+=1;self.statistics['proposed']+=len(proposals)
            self.statistics['accepted']+=min(len(committed)-1,len(values))
            if policy:
                policy.observe(depth,len(proposals),len(committed)-1,(time.perf_counter()-begin)*1000)
                self.statistics.setdefault('depth_rounds',{})[depth]=self.statistics.get('depth_rounds',{}).get(depth,0)+1
            generated.extend(values)
            if self.target.position!=self.draft.position:raise ValueError('MTP cursor diverged')
            if reason is not None:break
        return generated,reason or 'length'

    def snapshot(self, *, cpu=False):
        if self.hidden is None or self.target.position!=self.draft.position:raise ValueError('No complete session')
        return {'target':self.target.snapshot(cpu=cpu),'draft':self.draft.snapshot(cpu=cpu),
                'hidden':self.hidden.to(device='cpu',copy=True) if cpu else self.hidden.clone(),
                'pending':self.pending,'draft_token':self.draft_token}

    def restore(self, saved):
        if not isinstance(saved,dict) or set(saved)!={'target','draft','hidden','pending','draft_token'}:
            raise ValueError('Invalid MTP session snapshot')
        for key,model in (('target',self.target),('draft',self.draft)):
            validate_snapshot(saved[key],model.states,model.capacity,model.V,model.ple.ngram-1,
                              prefix_divisors=model.prefix_divisors,allow_cpu=True)
            if model.ple.eos in saved[key]['history']:raise ValueError('Invalid private history')
        hidden=saved['hidden']
        if not isinstance(hidden,torch.Tensor) or hidden.shape!=(1,self.target.C,self.target.H) or hidden.dtype!=torch.float16 or str(hidden.device) not in ('cpu',str(self.target.position_gpu.device)):
            raise ValueError('Invalid MTP HC condition')
        if saved['target']['position']!=saved['draft']['position'] or saved['draft']['history']:
            raise ValueError('MTP snapshot cursors or history differ')
        if any(type(saved[k]) is not int or not 0<=saved[k]<self.target.V for k in ('pending','draft_token')):
            raise ValueError('Invalid pending or draft token')
        # The cache may have been created with another draft-only vocabulary.
        # Recompute before mutating either private state, preserving atomicity.
        restored_hidden=hidden.to(self.target.position_gpu.device,copy=True)
        prediction=self.draft.head_token(restored_hidden)
        self.target.restore(saved['target']);self.draft.restore(saved['draft'])
        self.hidden=restored_hidden
        self.pending=saved['pending'];self.draft_token=prediction
        self.statistics={'rounds':0,'proposed':0,'accepted':0,'fallback_steps':0}
