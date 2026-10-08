from uni3r.utils.path_manager import init_all_submodules
init_all_submodules()
import torch
from uni3r.lseg import LSegFeatureExtractor
model=LSegFeatureExtractor.from_pretrained('/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R/checkpoints/demo_e200.ckpt').cuda().eval()
x=torch.zeros(1,3,256,256,device='cuda')
with torch.no_grad():
 y=model.extract_features(x)
 z=model.decode_feature(y,labelset=['wall','floor','ceiling','chair','table','sofa','bed','other'])
assert torch.isfinite(y).all() and torch.isfinite(z).all()
print('LSeg smoke passed:',y.shape,z.shape,flush=True)
