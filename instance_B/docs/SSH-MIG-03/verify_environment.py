import json, os, sys, pathlib, importlib.metadata as md
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import torch, numpy as np, scipy, pytest
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert not torch.cuda.is_initialized()
assert torch.__version__ == "2.1.0+cu121"
assert np.__version__ == "1.26.4" and scipy.__version__ == "1.15.3"
assert sys.version_info[:2] == (3,10)
a=np.arange(6,dtype=np.float32).reshape(2,3)
t=torch.from_numpy(a)
assert t.device.type == "cpu"
np.testing.assert_array_equal((t+1).numpy(),a+1)
assert (torch.tensor([[1.,2.]],device="cpu") @ torch.tensor([[3.],[4.]],device="cpu")).item() == 11
from instance_B import matcher, criterion, loss_utils
assert not torch.cuda.is_initialized()
r={"python":sys.version,"executable":sys.executable,"prefix":sys.prefix,"base_prefix":sys.base_prefix,"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"numpy":np.__version__,"scipy":scipy.__version__,"pytest":pytest.__version__,"CPU_tensor_math":"PASS","numpy_torch_roundtrip":"PASS","core_imports":"PASS","CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],"cuda_initialized":torch.cuda.is_initialized(),"GPU_USED":"NO","packages":{x.metadata["Name"]:x.version for x in md.distributions()}}
pathlib.Path(__file__).with_name("environment_versions.json").open("x").write(json.dumps(r,indent=2)+"\n")
print(json.dumps(r,indent=2))
