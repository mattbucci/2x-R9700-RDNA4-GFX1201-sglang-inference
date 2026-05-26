import json,torch,safetensors.torch as st
from pathlib import Path
M=Path("/data/cache/huggingface/hub/models--mattbucci--Qwen3-Coder-30B-A3B-REAM-AWQ");snap=next((M/"snapshots").glob("*"))
idx=json.load(open(snap/"model.safetensors.index.json"))["weight_map"]
def get(P):
 t=st.load_file(snap/idx[P+".qweight"]);return t[P+".qweight"].cuda(),t[P+".qzeros"].cuda(),t[P+".scales"].cuda()
order=[0,4,1,5,2,6,3,7]
def dq(qw,qz,sc,gs=128):
 K,Np=qw.shape;N=Np*8;w=torch.zeros(K,N,dtype=torch.int32,device='cuda');z=torch.zeros(qz.shape[0],N,dtype=torch.int32,device='cuda')
 for i,o in enumerate(order): w[:,i::8]=(qw>>(4*o))&0xF;z[:,i::8]=(qz>>(4*o))&0xF
 return ((w-z.repeat_interleave(gs,0))*sc.repeat_interleave(gs,0)).float()  # (K,N)
e="model.layers.0.mlp.experts.0"
g=dq(*get(e+".gate_proj"));u=dq(*get(e+".up_proj"));d=dq(*get(e+".down_proj"))
x=torch.randn(1,g.shape[0],device='cuda')  # hidden = K of gate = 2048
gate=x@g;up=x@u;h=torch.nn.functional.silu(gate)*up;y=h@d
print("gate",gate.std().item(),"up",up.std().item(),"h",h.std().item(),"y",y.std().item(),"absmax",y.abs().max().item())
print("expert0 full MLP:", "SANE" if 0.01<y.std()<50 and not torch.isnan(y).any() else "BROKEN")
