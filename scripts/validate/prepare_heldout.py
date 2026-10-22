"""Freeze all100 public held-out rows using unchanged reference preprocessing."""
import os,sys,json,hashlib,datetime
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path(os.environ['BAGEL_EVAL_DIR']).resolve();ROOT.mkdir(parents=True,exist_ok=True);os.chdir(ROOT);sys.path.insert(0,os.environ['BAGEL_REF_TREE'])
import torch,yaml
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig,PackedDataset
from data.vlm_dataset import SftJSONLIterableDataset
from data.transforms import ImageTransform
from data.video_utils import FrameSampler
from data.dataset_info import DATASET_INFO
from modeling.qwen2 import Qwen2Tokenizer
print('prepare_pid',os.getpid(),flush=True)
held=Path(os.environ['BAGEL_HELDOUT_JSONL']);train=Path(os.environ['BAGEL_TRAIN_JSONL'])
rows=[json.loads(x) for x in held.read_text().splitlines()];assert len(rows)==100
images=Path(DATASET_INFO['vlm_sft']['llava_ov']['data_dir'])
def image_paths(rows):
    out=set()
    for r in rows:
        assert 'video' not in r
        names=r.get('image',[]);names=[names] if isinstance(names,str) else names
        out.update(images/n for n in names)
    return out
train_images=image_paths([json.loads(x) for x in train.read_text().splitlines()]);held_images=image_paths(rows)
def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
train_hashes={file_sha(p) for p in train_images};held_hashes={file_sha(p) for p in held_images};assert not (train_hashes&held_hashes)
args=yaml.safe_load(Path(os.environ['DATASET_CONFIG']).read_text())['vlm_sft']
tok=Qwen2Tokenizer.from_pretrained(os.environ['MODEL_PATH']);tok,special,added=add_special_tokens(tok)
ds=SftJSONLIterableDataset('heldout',ImageTransform(**args['image_transform_args']),tok,FrameSampler(**args['frame_sampler_args']),[str(held)],[str(images)],[100],local_rank=0,world_size=1,num_workers=1,shuffle_lines=False)
# The training iterable normally shuffles in set_epoch; evaluation keeps JSONL order.
ds.data_paths_per_rank=ds.data_paths;ds.num_files_per_rank=100
packer=PackedDataset(DataConfig({},text_cond_dropout_prob=0,vit_cond_dropout_prob=0,vae_cond_dropout_prob=0,max_latent_size=64),tok,special,0,1,1,expected_num_tokens=4096,max_num_tokens_per_sample=4096,max_num_tokens=6144,prefer_buffer_before=4096,use_flex=True)
out=Path('heldout_batches');out.mkdir(exist_ok=True);manifest=[]
iterator=iter(ds)
for i in range(100):
    sample=next(iterator);assert sample['data_indexes']['data_indexes']==i,('skipped row',i,sample['data_indexes'])
    status=packer.pack_sequence(sample,packer.set_sequence_status())
    assert status is not None
    n=sum(status['sample_lens']);assert n<=6144,('heldout exceeds capacity; must handle without truncation',i,n)
    data=packer.to_tensor(status);data.pop('ce_loss_weights',None)
    assert data['packed_label_ids'].numel()>0 and data['vit_token_seqlens'].numel()>0
    assert 'packed_timesteps' not in data and 'mse_loss_indexes' not in data
    p=out/f'{i:03d}.pt';torch.save(data,p)
    manifest.append({'row':i,'batch':str(p),'sha256':file_sha(p),'sequence_length':n,'ce_tokens':data['packed_label_ids'].numel(),'images':data['vit_token_seqlens'].numel()})
report={'status':'PASS','pid':os.getpid(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':100,'source':str(held),'source_sha256':file_sha(held),'training_source_sha256':file_sha(train),'train_image_content_count':len(train_hashes),'heldout_image_content_count':len(held_hashes),'image_content_overlap':0,'ordering':'heldout JSONL source order, no shuffle or omitted row','preprocessing':'unchanged reference SftJSONLIterableDataset/ImageTransform/PackedDataset.pack_sequence and to_tensor; same VLM transform; all conditioning dropout probabilities0','capacity':6144,'tokenizer_length':len(tok),'special_tokens':special,'total_ce_tokens':sum(r['ce_tokens'] for r in manifest),'batches':manifest}
Path('heldout_manifest.json').write_text(json.dumps(report,indent=2)+'\n');assert not torch.cuda.is_initialized();print('prepared100rows',report['total_ce_tokens'],flush=True)
