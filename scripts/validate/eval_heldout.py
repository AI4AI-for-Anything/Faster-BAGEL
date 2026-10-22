"""Common reference-code teacher-forced held-out CE, used unchanged for every arm."""
import argparse, datetime, gc, hashlib, json, math, os, sys, time
from pathlib import Path
ROOT=Path(os.environ['BAGEL_EVAL_DIR']).resolve()
REF=Path(os.environ['BAGEL_REF_TREE']).resolve()
sys.path.insert(0,str(REF))
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import set_seed
from data.data_utils import add_special_tokens
from modeling.autoencoder import AutoEncoderParams
from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer
from train.fsdp_utils import FSDPConfig, fsdp_wrapper

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def to_device(x,device):
    if isinstance(x,torch.Tensor):return x.to(device)
    if isinstance(x,list):return [to_device(y,device) for y in x]
    if isinstance(x,tuple):return tuple(to_device(y,device) for y in x)
    if isinstance(x,dict):return {k:to_device(v,device) for k,v in x.items()}
    return x

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==8
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    (out/f'rank{rank}.pid').write_text(str(os.getpid())+'\n')
    start=time.time();set_seed(12345)
    assert not any(k.startswith('BAGEL_OPT_') for k in os.environ)
    manifest=json.loads((ROOT/'heldout_manifest.json').read_text())
    assert manifest['rows']==100 and manifest['image_content_overlap']==0
    base=Path(os.environ['MODEL_PATH'])
    llm=Qwen2Config.from_json_file(str(base/'llm_config.json'))
    llm.layer_module='Qwen2MoTDecoderLayer';llm.qk_norm=True;llm.tie_word_embeddings=False;llm.freeze_und=False
    language=Qwen2ForCausalLM(llm)
    vit=SiglipVisionConfig.from_json_file(str(base/'vit_config.json'));vit.num_hidden_layers+=1-2;vit.rope=False
    vision=SiglipVisionModel(vit)
    vae=AutoEncoderParams(resolution=256,in_channels=3,downsample=8,ch=128,out_ch=3,ch_mult=[1,2,4,4],num_res_blocks=2,z_channels=16,scale_factor=0.3611,shift_factor=0.1159)
    config=BagelConfig(visual_gen=True,visual_und=True,llm_config=llm,vit_config=vit,vae_config=vae,latent_patch_size=2,max_latent_size=64,vit_max_num_patch_per_side=70,connector_act='gelu_pytorch_tanh',interpolate_pos=False,timestep_shift=1.0)
    model=Bagel(language,vision,config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit)
    tokenizer=Qwen2Tokenizer.from_pretrained(str(base));tokenizer,_,added=add_special_tokens(tokenizer)
    if added:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size=len(tokenizer);model.language_model.config.vocab_size=len(tokenizer)
    assert len(tokenizer)==manifest['tokenizer_length']
    # Full strict load includes both experts and fixed positional embeddings. No model surgery.
    for load_rank in range(world):
        if rank==load_rank:
            print(f'rank={rank} loading {args.checkpoint}',flush=True)
            state=load_file(args.checkpoint,device='cpu')
            assert all(v.dtype==torch.float32 for v in state.values() if v.is_floating_point())
            model.load_state_dict(state,strict=True);del state;gc.collect()
        dist.barrier()
    checkpoint_hash=sha(args.checkpoint) if rank==0 else None
    # The reference dispatches to autoregressive inference when .training=False.
    # Bind its unchanged packed teacher-forced methods while retaining eval mode everywhere.
    dispatch=[]
    model.eval()
    for name,module in model.named_modules():
        if module.__class__.__module__=='modeling.bagel.qwen2_navit' and hasattr(module,'forward_train'):
            module.forward=module.forward_train;dispatch.append(name)
    assert dispatch and all(not m.training for m in model.modules())
    # Held-out rows contain only VLM; skip the branch requiring VAE/flow inputs.
    # Full generation parameters were loaded above and remain present in the model.
    model.config.visual_gen=False
    for param in model.parameters():param.requires_grad_(False)
    fsdp=fsdp_wrapper(model,FSDPConfig('HYBRID_SHARD','BACKWARD_PRE',False,1,8));fsdp.eval()
    assert all(not m.training for m in fsdp.modules())
    local=[]
    with torch.no_grad():
        for step in range((100+world-1)//world):
            index=step*world+rank;real=index<100
            row=manifest['batches'][index if real else 0]
            path=ROOT/row['batch'];assert sha(path)==row['sha256']
            batch=torch.load(path,map_location='cpu',weights_only=True)
            assert batch['packed_label_ids'].numel()==row['ce_tokens']
            batch=to_device(batch,device)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):loss=fsdp(**batch)['ce']
            assert loss.numel()==row['ce_tokens']
            finite=torch.isfinite(loss);bad=int((~finite).sum().item())
            loss_sum=float(loss.double().sum().item())
            record={'row':index,'rank':rank,'ce_tokens':loss.numel(),'loss_sum':loss_sum if math.isfinite(loss_sum) else None,'mean_ce':loss_sum/loss.numel() if math.isfinite(loss_sum) else None,'nonfinite_tokens':bad,'finite':bad==0}
            if real:
                local.append(record)
                with (out/f'rows_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(f'rank={rank} row={index} included={real} ce={loss_sum/loss.numel()} finite={bad==0}',flush=True)
            del loss,batch
    gathered=[None]*world if rank==0 else None
    dist.gather_object(local,gathered,dst=0)
    if rank==0:
        rows=sorted([r for part in gathered for r in part],key=lambda r:r['row'])
        assert [r['row'] for r in rows]==list(range(100))
        tokens=sum(r['ce_tokens'] for r in rows);assert tokens==manifest['total_ce_tokens']
        finite=all(r['finite'] for r in rows)
        result={'status':'ok' if finite else 'nonfinite','rows':100,'total_ce_tokens':tokens,'mean_text_ce':math.fsum(r['loss_sum'] for r in rows)/tokens if finite else None,'mean_row_ce':math.fsum(r['mean_ce'] for r in rows)/100 if finite else None,'all_finite':finite,'nonfinite_rows':[r['row'] for r in rows if not r['finite']],'per_row':rows,'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_bytes':Path(args.checkpoint).stat().st_size,'checkpoint_sha256':checkpoint_hash,'evaluator_sha256':sha(__file__),'manifest_sha256':sha(ROOT/'heldout_manifest.json'),'reference_tree':str(REF),'precision':'FP32 checkpoint parameters; reference FSDP BF16 mixed precision; torch.amp.autocast BF16','dropout':'all modules eval; conditioning dropout zero in frozen preprocessing','ordering':'rows 0..99; step*8+rank; four excluded dummy forwards for collective parity','dispatch':'reference packed forward_train bound locally with training=False','dispatch_modules':dispatch,'seed':12345,'wall_s':time.time()-start,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('RESULT',result['status'],result['mean_text_ce'],flush=True)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
