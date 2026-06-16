"""Validate the sync-free GPU-vectorized reconstruct_indices_from_tree_mask against a kernel-exact
python reference on random draft trees. Gate before putting the vectorized version in the live tree."""
import torch

def reference(tree_mask, vsl, bs, D):
    # exact port of ngram_utils.cu reconstructIndicesFromTreeMask
    tm = tree_mask.cpu().bool().view(bs, D, D).tolist()
    vsl = vsl.cpu().tolist()
    pos = [0]*(bs*D); ridx=[0]*(bs*D); rnt=[0]*(bs*D); rns=[0]*(bs*D)
    for bid in range(bs):
        tok = bid*D
        for tid in range(D):
            depth=0; parent=-1
            for i in range(tid-1,-1,-1):
                if tm[bid][tid][i]:
                    depth+=1
                    if parent==-1: parent=i
            ridx[tok+tid]=tok+tid
            pos[tok+tid]=depth+vsl[bid]
            nt=-1
            for i in range(tid+1,D):
                if tm[bid][i][tid]: nt=i; break
            rnt[tok+tid]=nt
            ns=-1
            if parent!=-1:
                for i in range(tid+1,D):
                    if tm[bid][i][parent]:
                        sib=True
                        for j in range(parent+1,i):
                            if tm[bid][i][j]: sib=False; break
                        if sib: ns=i; break
            rns[tok+tid]=ns
    return pos, ridx, rnt, rns

def vectorized(tree_mask, verified_seq_len, bs, D, dev):
    tm = tree_mask.view(bs, D, D).bool()
    vsl = verified_seq_len.view(bs).to(torch.int64)
    r = torch.arange(D, device=dev)
    idx = r.view(1,1,D).expand(bs,D,D).to(torch.int64)
    lt = (r.view(1,D) < r.view(D,1)).view(1,D,D)
    gt = (r.view(1,D) > r.view(D,1)).view(1,D,D)
    neg1 = torch.full((bs,D,D), -1, device=dev, dtype=torch.int64)
    big = torch.full((bs,D,D), D, device=dev, dtype=torch.int64)
    depth = (tm & lt).sum(-1).to(torch.int64)
    positions = (depth + vsl.view(bs,1)).view(-1)
    retrive_index = torch.arange(bs*D, device=dev, dtype=torch.int64)
    parent = torch.where(tm & lt, idx, neg1).amax(-1)
    nt = torch.where(tm.transpose(1,2) & gt, idx, big).amin(-1)
    next_token = torch.where(nt==D, neg1[:,:,0], nt).view(-1)
    sib = (parent.view(bs,1,D)==parent.view(bs,D,1)) & gt & (parent.view(bs,D,1)!=-1)
    sm = torch.where(sib, idx, big).amin(-1)
    next_sibling = torch.where(sm==D, neg1[:,:,0], sm).view(-1)
    return positions, retrive_index, next_token, next_sibling

def rand_tree_mask(bs, D, gen):
    # ancestor masks from random parent trees (token t>0 has parent < t)
    tm = torch.zeros(bs, D, D, dtype=torch.bool)
    for b in range(bs):
        parent = [-1]*D
        for t in range(1, D):
            parent[t] = int(torch.randint(0, t, (1,), generator=gen).item())
        for t in range(D):
            i = t
            while i != -1:
                tm[b, t, i] = True
                i = parent[i]
    return tm.view(-1)

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(1234)
    fails = 0
    for trial in range(300):
        bs = int(torch.randint(1,3,(1,),generator=gen).item())
        D = int(torch.randint(2,9,(1,),generator=gen).item())
        tm = rand_tree_mask(bs, D, gen)
        vsl = torch.randint(0, 100000, (bs,), generator=gen, dtype=torch.int64)
        rp, rr, rnt, rns = reference(tm, vsl, bs, D)
        tm_d = tm.to(dev); vsl_d = vsl.to(dev)
        vp, vr, vnt, vns = (x.cpu().tolist() for x in vectorized(tm_d, vsl_d, bs, D, dev))
        if [vp,vr,vnt,vns] != [rp,rr,rnt,rns]:
            fails += 1
            if fails <= 3:
                print(f"MISMATCH trial {trial} bs={bs} D={D}")
                print(" pos", rp, vp); print(" nt ", rnt, vnt); print(" ns ", rns, vns)
    print(f"device={dev}  {300-fails}/300 trials match" + (" — ALL PASS" if fails==0 else f" — {fails} FAIL"))

main()
