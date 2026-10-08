"""Expose LSeg backbone initialization control without changing checkpoint inference."""
from pathlib import Path
import difflib
root = Path(__file__).resolve().parents[1]
p = root/'submodules/lang_seg/modules/models/lseg_net.py'
old = '            features,\n            groups=1,\n'
new = '            features,\n            use_pretrained=kwargs.get("use_pretrained", True),\n            groups=1,\n'
s = p.read_text()
if new not in s:
    if s.count(old) != 1:
        raise RuntimeError('Unexpected LSeg source version; inspect before patching')
    updated = s.replace(old, new)
    patch = ''.join(difflib.unified_diff(s.splitlines(True), updated.splitlines(True),
                    fromfile='a/modules/models/lseg_net.py', tofile='b/modules/models/lseg_net.py'))
    (root/'reproduce/lseg_vendor.patch').write_text(patch)
    p.write_text(updated)
print('LSeg optional backbone initialization patch applied')
