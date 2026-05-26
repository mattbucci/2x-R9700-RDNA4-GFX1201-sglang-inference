import os,json,torch,safetensors.torch as st
from pathlib import Path
import sglang.srt.server_args as SA
class _S:
 def __getattr__(s,k): return False
SA._global_server_args=_S()
try: SA.set_global_server_args(_S())
except: pass
M=Path("/data/cache/huggingface/hub/models--mattbucci--Qwen3-Coder-30B-A3B-REAM-AWQ");snap=next((M/"snapshots").glob("*"))
idx=json.load(open(snap/"model.safetensors.index.json"))["weight_map"]
def g(P):t=st.load_file(snap/idx[P+".qweight"]);return t[P+".qweight"].cuda(),t[P+".qzeros"].cuda(),t[P+".scales"].cuda()
ord8=[0,4,1,5,2,6,3,7]
def dq(qw,qz,sc,gs=128):
 K,Np=qw.shape;N=Np*8;w=torch.zeros(K,N,dtype=torch.int32,device='cuda');z=torch.zeros(qz.shape[0],N,dtype=torch.int32,device='cuda')
 for i,o in enumerate(ord8):w[:,i::8]=(qw>>(4*o))&0xF;z[:,i::8]=(qz>>(4*o))&0xF
 return ((w-z.repeat_interleave(gs,0))*sc.repeat_interleave(gs,0)).float()
def conv(t):
 s0=t.size(0);t=t.view(torch.uint8);t=(t[:,:,None]>>torch.tensor([0,4],dtype=torch.uint8,device='cuda'))&0xF
 t=t.view(-1,8)[:,ord8];t=t.view(s0,-1).T.contiguous();return t[:,1::2]*16+t[:,::2]
def cz(z):z=z.view(torch.uint8);z=(z[:,:,None]>>torch.tensor([0,4],dtype=torch.uint8,device='cuda'))&0xF;z=z.view(-1,8)[:,ord8].view(z.size(0),-1).T.contiguous();return z[1::2]*16+z[::2]
EX=[77,36,1,41,74,87,14,46]
x=torch.randn(1,2048,device='cuda',dtype=torch.float16);tw=torch.softmax(torch.tensor([0.087,0.11,0.115,0.12,0.13,0.14,0.15,0.15]),0).cuda()[None]
ti=torch.arange(8,device='cuda')[None]
w13=[];w2=[];s13=[];s2=[];z13=[];z2=[];refsum=0
for j,e in enumerate(EX):
 P=f"model.layers.0.mlp.experts.{e}";gq,uq,dd=g(P+".gate_proj"),g(P+".up_proj"),g(P+".down_proj")
 gg,uu,d=dq(*gq),dq(*uq),dq(*dd);h=torch.nn.functional.silu(x.float()@gg)*(x.float()@uu);refsum=refsum+tw[0,j].item()*(h@d)
 w13.append(torch.cat([conv(gq[0]),conv(uq[0])],0));w2.append(conv(dd[0]));s13.append(torch.cat([gq[2],uq[2]],1).T);s2.append(dd[2].T);z13.append(torch.cat([cz(gq[1]),cz(uq[1])],0));z2.append(cz(dd[1]))
w13=torch.stack(w13);w2=torch.stack(w2);s13=torch.stack(s13);s2=torch.stack(s2);z13=torch.stack(z13);z2=torch.stack(z2)
print("REFSUM std",refsum.std().item(),"max",refsum.abs().max().item())
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
y=fused_experts_impl(x,w13,w2,tw,ti,use_int4_w4a16=True,w1_scale=s13,w2_scale=s2,w1_zp=z13,w2_zp=z2,block_shape=[0,128])
print("8EXP KERNEL std",y.float().std().item(),"max",y.abs().max().item(),"ratio",y.float().std().item()/refsum.std().item())
