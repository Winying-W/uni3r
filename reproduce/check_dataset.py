from uni3r.utils.path_manager import init_all_submodules
init_all_submodules()
from uni3r.datasets.testdata import TestDataset
from pathlib import Path
root=Path('/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R/data_root.txt').read_text()
for n in (2,8,16):
 d=TestDataset(split='test',ROOT=root,resolution=(256,256),seed=777,num_views=n)
 b=d[0]
 print(n,'views, samples=',len(d),'scenes=',len(d.scenes),'batch_views=',len(b),'image=',b[0]['img'].shape,flush=True)
