"""Optional whole-document redistribution around the replicated MoT decoder.

Only activations move. Outputs and their backward gradients return to their
original rank/order, so the loader, losses, RNG and optimizer retain ownership.
"""
import bisect
import os

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single
from torch.nn.attention.flex_attention import create_block_mask

ENABLED = os.environ.get('BAGEL_OPT_LM_BALANCE', '') == '1'


def documents(sample_lens, split_lens, modes, valid, gen_ids):
    """Split public packing metadata at document boundaries, excluding padding."""
    if split_lens is None or modes is None or len(split_lens) != len(modes):
        return None
    cursor = seg = 0
    result = []
    gen_ids = sorted(gen_ids)
    for size in sample_lens:
        if cursor == valid:
            break
        if size <= 0 or cursor + size > valid:
            return None
        lengths, kinds, remaining = [], [], size
        while remaining:
            if seg >= len(split_lens) or not 0 < split_lens[seg] <= remaining:
                return None
            if modes[seg] not in ('causal', 'full', 'noise'):
                return None
            lengths.append(split_lens[seg]); kinds.append(modes[seg])
            remaining -= split_lens[seg]; seg += 1
        ngen = bisect.bisect_left(gen_ids, cursor+size)-bisect.bisect_left(gen_ids,cursor)
        result.append((cursor,size,ngen,lengths,kinds))
        cursor += size
    return result if cursor == valid and result else None


def placement(specs):
    """Deterministic capacity-limited greedy assignment; identity on no gain."""
    world = len(specs)
    entries = [(rank,i,d) for rank,s in enumerate(specs) for i,d in enumerate(s['docs'])]
    if len(entries) < world:
        return None
    capacity = max(s['capacity'] for s in specs)
    def work(d):
        _,n,g,_,_=d
        return (n,n-g,g,n*n)
    totals = [sum(work(d)[j] for _,_,d in entries) for j in range(4)]
    means = [max(t/world,1) for t in totals]
    def score(load):
        # Total tokens dominate GEMM work; expert skew and long attention
        # documents also matter. No model or dataset dimensions are assumed.
        values=[load[j]/means[j] for j in range(4)]
        return max(values[:3])+0.2*values[3]
    before=[[sum(work(d)[j] for d in s['docs']) for j in range(4)] for s in specs]
    loads=[[0]*4 for _ in specs]
    owners={}
    for index,(src,i,d) in enumerate(sorted(entries,key=lambda e:(-e[2][1],e[0],e[1]))):
        w=work(d)
        choices=[r for r in range(world) if loads[r][0]+w[0]<=capacity and (index>=world or loads[r][0]==0)]
        if not choices:
            return None
        rank=min(choices,key=lambda r:(score([a+b for a,b in zip(loads[r],w)]),loads[r][0],r))
        loads[rank]=[a+b for a,b in zip(loads[rank],w)]
        owners[src,i]=rank
    if max(map(score,loads)) >= max(map(score,before)):
        return None
    return owners


def route(specs, owners, rank):
    world=len(specs)
    send=[0]*world; receive=[0]*world; order=[]; received=[]
    for dst in range(world):
        for i,d in enumerate(specs[rank]['docs']):
            if owners[rank,i]==dst:
                begin,n,*_=d
                order.extend(range(begin,begin+n));send[dst]+=n
    for src,spec in enumerate(specs):
        for i,d in enumerate(spec['docs']):
            if owners[src,i]==rank:
                received.append(d);receive[src]+=d[1]
    inverse=[0]*len(order)
    for i,v in enumerate(order):inverse[v]=i
    return send,receive,order,inverse,received


def _exchange(x, send, receive):
    out=x.new_empty((sum(receive),*x.shape[1:]))
    return all_to_all_single(out,x.contiguous(),list(receive),list(send))


def try_forward(lm, sequence, sample_lens, split_lens, modes, mask, positions, extra):
    if not ENABLED or not lm.training or not torch.is_grad_enabled() or not dist.is_initialized() or dist.get_world_size()<2:
        return None
    cfg=lm.config
    supported=(getattr(lm.model,'use_moe',False) and not getattr(cfg,'freeze_und',False)
               and not getattr(cfg,'attention_dropout',0) and not getattr(cfg,'rope_scaling',None)
               and not isinstance(mask,list) and 'packed_und_token_indexes' in extra)
    gen=extra.get('packed_gen_token_indexes')
    if supported:
        supported = (extra['packed_und_token_indexes'].numel()
                     + (gen.numel() if gen is not None else 0) == sequence.shape[0])
    ids=gen.detach().cpu().tolist() if supported and gen is not None else []
    docs=documents(sample_lens,split_lens,modes,sequence.shape[0],ids) if supported else None
    spec={'docs':docs,'capacity':sum(sample_lens)} if docs else None
    specs=[None]*dist.get_world_size()
    dist.all_gather_object(specs,spec)
    if any(s is None for s in specs):
        return None
    owners=placement(specs)
    if owners is None:
        return None
    send,receive,order,inverse,received=route(specs,owners,dist.get_rank())
    from train.async_log import to_device
    def index(values):return to_device(torch.tensor(values,dtype=torch.long),sequence.device)
    si,inv=index(order),index(inverse)
    x=_exchange(sequence.index_select(0,si),send,receive)
    # Integer positions and expert membership use the same reversible route.
    metadata=torch.stack((positions,torch.zeros_like(positions)),dim=1)
    if gen is not None:metadata[gen,1]=1
    moved=metadata.new_empty((sum(receive),2))
    dist.all_to_all_single(moved,metadata.index_select(0,si).contiguous(),receive,send)
    lengths=[d[1] for d in received]
    splits=[v for d in received for v in d[3]]
    kinds=[v for d in received for v in d[4]]
    capacity=max(s['capacity'] for s in specs)
    padding=capacity-sum(lengths)
    if padding:
        lengths.append(padding);splits.append(padding);kinds.append('causal')
    from data.data_utils import create_sparse_mask
    sparse=create_sparse_mask(lengths,splits,kinds,sequence.device)
    balanced_mask=create_block_mask(sparse,B=1,H=lm.model.config.num_attention_heads,
                                   Q_LEN=capacity,KV_LEN=capacity,device=sequence.device,BLOCK_SIZE=128,_compile=True)
    und=torch.where(moved[:,1]==0)[0]
    gen=torch.where(moved[:,1]==1)[0]
    y=lm(packed_sequence=x,sample_lens=lengths,attention_mask=balanced_mask,
         packed_position_ids=moved[:,0],packed_und_token_indexes=und,packed_gen_token_indexes=gen)
    lm._bagel_lm_balance_calls=getattr(lm,'_bagel_lm_balance_calls',0)+1
    return _exchange(y,receive,send).index_select(0,inv)
