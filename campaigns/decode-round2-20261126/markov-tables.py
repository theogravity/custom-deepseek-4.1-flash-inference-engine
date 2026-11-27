from pathlib import Path
import json,struct,hashlib,array,sys
p=Path('/mnt/models/dsv41-flash-exl3/drafter-code-k7')
idx=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
out={}
for name in ['mtp.2.markov_head.embed.weight','mtp.2.markov_head.head.weight']:
 f=p/idx[name]
 with f.open('rb') as stream:
  n=struct.unpack('<Q',stream.read(8))[0];t=json.loads(stream.read(n))[name]
 assert t['dtype']=='BF16' and t['shape']==[129280,256]
 start,end=t['data_offsets']; assert end-start==2*129280*256
 with f.open('rb') as stream:
  stream.seek(8+n+start); data=stream.read(end-start)
 w=array.array('H');w.frombytes(data)
 if sys.byteorder!='little':w.byteswap()
 hi=max(v & 0x7fff for v in w);assert hi<=0x4f80
 out[name]={'source':str(f),'shape':t['shape'],'bf16_max_magnitude_bits':hex(hi),'required_max_magnitude_bits':'0x4f80','sha256':hashlib.sha256(data).hexdigest()}
(Path(__file__).resolve().parent/'markov/table-bounds.json').write_text(json.dumps(out,indent=2)+'\n')
