import torch
from simple_knn._C import distCUDA2
from gsplat import rasterization
torch.manual_seed(777)
x=torch.randn(128,3,device='cuda')
d=torch.cdist(x,x).square();d.fill_diagonal_(float('inf'))
expected=d.topk(3,largest=False).values.mean(1)
actual=distCUDA2(x)
torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
means=torch.tensor([[0.,0.,3.]],device='cuda',requires_grad=True)
quats=torch.tensor([[1.,0.,0.,0.]],device='cuda')
scales=torch.full((1,3),0.1,device='cuda')
opacities=torch.tensor([0.8],device='cuda')
colors=torch.tensor([[1.,0.2,0.1]],device='cuda')
K=torch.tensor([[[50.,0.,16.],[0.,50.,16.],[0.,0.,1.]]],device='cuda')
view=torch.eye(4,device='cuda')[None]
r,a,_=rasterization(means,quats,scales,opacities,colors,view,K,32,32)
assert torch.isfinite(r).all() and a.max()>0
r.sum().backward()
assert torch.isfinite(means.grad).all()
print('L20 nearest-neighbor and Gaussian rendering/backward checks passed',flush=True)
print('Peak MiB',torch.cuda.max_memory_allocated()/2**20,flush=True)
