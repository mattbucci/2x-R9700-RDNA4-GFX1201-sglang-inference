import json,torch,safetensors.torch as st
from pathlib import Path
M=Path("/data/cache/huggingface/hub/models--mattbucci--Qwen3-Coder-30B-A3B-REAM-AWQ");snap=next((M/"snapshots").glob("*"))
idx=json.load(open(snap/"model.safetensors.index.json"))["weight_map"];P="model.layers.0.mlp.experts.0.gate_proj"
ts=st.load_file(snap/idx[P+".qweight"]);qw,qz,sc=ts[P+".qweight"].cuda(),ts[P+".qzeros"].cuda(),ts[P+".scales"].cuda()
order=[0,4,1,5,2,6,3,7]
def casper(qw,qz,sc,gs=128):
 K,Np=qw.shape;N=Np*8;w=torch.zeros(K,N,dtype=torch.int32,device='cuda');z=torch.zeros(qz.shape[0],N,dtype=torch.int32,device='cuda')
 for i,o in enumerate(order): w[:,i::8]=(qw>>(4*o))&0xF;z[:,i::8]=(qz>>(4*o))&0xF
 return ((w-z.repeat_interleave(gs,0))*sc.repeat_interleave(gs,0)), w
ref,wint=casper(qw,qz,sc)               # ref: (K,N) dequant ; wint: (K,N) int4
def conv(t):
 s0=t.size(0);t=t.view(torch.uint8);t=(t[:,:,None]>>torch.tensor([0,4],dtype=torch.uint8,device='cuda'))&0xF
 t=t.view(-1,8)[:,order];t=t.view(s0,-1).T.contiguous();return t[:,1::2]*16+t[:,::2]
pqw=conv(qw)                            # (N, K//2)
lo=(pqw&0xF).to(torch.int32);hi=(pqw>>4).to(torch.int32)
N,K2=pqw.shape;w=torch.zeros(N,K2*2,dtype=torch.int32,device='cuda');w[:,0::2]=lo;w[:,1::2]=hi
print("prod ints == casper ints:",torch.equal(w, wint.T))
