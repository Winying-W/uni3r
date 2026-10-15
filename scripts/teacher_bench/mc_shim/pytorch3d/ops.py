"""Minimal torch stand-in for pytorch3d.ops.ball_query (the only pytorch3d op MaskClustering
uses). Semantics as pytorch3d: for each query point, indices of up to K points of p2 within
`radius`, in increasing index order, padded with -1. MaskClustering only uses idx."""
import torch


def ball_query(p1, p2, lengths1=None, lengths2=None, K=500, radius=0.2, return_nn=False):
    N, P1, _ = p1.shape
    P2 = p2.shape[1]
    idx = torch.full((N, P1, K), -1, dtype=torch.long, device=p1.device)
    r2 = radius * radius
    for n in range(N):
        l1 = int(lengths1[n]) if lengths1 is not None else P1
        l2 = int(lengths2[n]) if lengths2 is not None else P2
        if l1 == 0 or l2 == 0:
            continue
        b = p2[n, :l2]
        chunk = max(1, min(4096, (64 * 1024 * 1024) // max(l2, 1)))  # <=64M entries per block
        for s in range(0, l1, chunk):
            a = p1[n, s:min(s + chunk, l1)]
            within = torch.cdist(a, b).pow(2) <= r2            # (q, l2)
            rank = within.cumsum(1)                              # 1-based rank among hits
            keep = within & (rank <= K)
            q, j = keep.nonzero(as_tuple=True)
            idx[n, s + q, rank[q, j] - 1] = j
    return None, idx, None
