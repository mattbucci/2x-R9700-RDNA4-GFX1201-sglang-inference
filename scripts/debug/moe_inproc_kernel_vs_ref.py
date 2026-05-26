import os,json,torch,safetensors.torch as st
from pathlib import Path
M=Path("/data/cache/huggingface/hub/models--mattbucci--Qwen3-Coder-30B-A3B-REAM-AWQ");snap=next((M/"snapshots").glob("*"))
idx=json.load(open(snap/"model.safetensors.index.json"))["weight_map"]
def g(P):t=st.load_file(snap/idx[P+".qweight"]);return t[P+".qweight"].cuda(),t[P+".qzeros"].cuda(),t[P+".scales"].cuda()
ord8=[0,4,1,5,2,6,3,7]
def dq(qw,qz,sc,gs=128):
 K,Np=qw.shape;N=Np*8;w=torch.zeros(K,N,dtype=torch.int32,device='cuda');z=torch.zeros(qz.shape[0],N,dtype=torch.int32,device='cuda')
 for i,o in enumerate(ord8):w[:,i::8]=(qw>>(4*o))&0xF;z[:,i::8]=(qz>>(4*o))&0xF
 return ((w-z.repeat_interleave(gs,0))*sc.repeat_interleave(gs,0)).float()
e="model.layers.0.mlp.experts.0";gq=g(e+".gate_proj");uq=g(e+".up_proj");dq2=g(e+".down_proj")
gg,uu,dd=dq(*gq),dq(*uq),dq(*dq2)
x=torch.randn(1,2048,device='cuda',dtype=torch.float16);h=torch.nn.functional.silu(x.float()@gg)*(x.float()@uu);ref=(h@dd)
print("REF std",ref.std().item(),"max",ref.abs().max().item())
def conv(t):
 s0=t.size(0);t=t.view(torch.uint8);t=(t[:,:,None]>>torch.tensor([0,4],dtype=torch.uint8,device='cuda'))&0xF
 t=t.view(-1,8)[:,ord8];t=t.view(s0,-1).T.contiguous();return t[:,1::2]*16+t[:,::2]
w13=torch.stack([torch.cat([conv(gq[0]),conv(uq[0])],0)]).contiguous();w2=conv(dq2[0])[None].contiguous()
s13=torch.cat([gq[2],uq[2]],1).T[None].contiguous();s2=dq2[2].T[None].contiguous()
def cz(z):z=z.view(torch.uint8);z=(z[:,:,None]>>torch.tensor([0,4],dtype=torch.uint8,device='cuda'))&0xF;z=z.view(-1,8)[:,ord8].view(z.size(0),-1).T.contiguous();return z[1::2]*16+z[::2]
z13=torch.cat([cz(gq[1]),cz(uq[1])],0)[None].contiguous();z2=cz(dq2[1])[None].contiguous()

import sglang.srt.server_args as SA
class _Stub:
 def __getattr__(s,k): return False
SA._global_server_args=_Stub()
try: SA.set_global_server_args(_Stub())
except: pass
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
tw=torch.ones(1,1,device='cuda');ti=torch.zeros(1,1,dtype=torch.long,device='cuda')
y=fused_experts_impl(x,w13,w2,tw,ti,activation="silu",use_int4_w4a16=True,w1_scale=s13,w2_scale=s2,w1_zp=z13,w2_zp=z2,block_shape=[0,128])
print("KERNEL std",y.float().std().item(),"max",y.abs().max().item(),"ratio",y.float().std().item()/ref.std().item())
