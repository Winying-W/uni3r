"""Extract ONLY DPT storage records; never unpickle the original large checkpoint.

This narrowly supported protocol-2 parser treats GLOBAL/REDUCE/NEWOBJ/BUILD as
data descriptions. It never imports/calls any checkpoint-specified function.
Unsupported opcodes/globals/formats fail closed. Not a general pickle loader.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickletools
import sys
import time
import zipfile

import torch

from instance_B.dpt_adapter import ROOT, make_reference_dpt, strict_load_dpt

BASE_SHA256 = "660aed13d6a8bccd078be35210f803d0f6d3b312f63dd468bbccc1c7f3aacf24"
BASE_SIZE = 11913939138
BASE_MTIME_NS = 1791421891990014631
GLOBALS = {"argparse Namespace", "collections OrderedDict", "torch FloatStorage",
           "torch HalfStorage", "torch LongStorage", "torch._utils _rebuild_tensor_v2"}


@dataclass(frozen=True)
class GlobalName:
    name: str


@dataclass(frozen=True)
class StorageSpec:
    kind: str
    key: str
    numel: int


@dataclass(frozen=True)
class TensorSpec:
    storage: StorageSpec
    offset: int
    shape: tuple
    stride: tuple


class StaticMap(dict):
    """Inert mapping, including ignored Namespace/state_dict attributes."""
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.attributes = None


def parse_tensor_descriptions(raw: bytes):
    if len(raw) > 4 * 1024**2:
        raise ValueError("Checkpoint pickle metadata exceeds inspection limit")
    stack, memo, marker = [], {}, object()
    def marked():
        i = len(stack) - 1
        while i >= 0 and stack[i] is not marker:
            i -= 1
        if i < 0:
            raise ValueError("Malformed MARK")
        values = stack[i+1:]
        del stack[i:]
        return values
    for count, (op, arg, _) in enumerate(pickletools.genops(raw)):
        if count > 250000:
            raise ValueError("Too many metadata instructions")
        name = op.name
        if name == "PROTO":
            if arg != 2:
                raise ValueError("Only inspected protocol 2 is supported")
        elif name == "EMPTY_DICT": stack.append({})
        elif name == "EMPTY_TUPLE": stack.append(())
        elif name == "MARK": stack.append(marker)
        elif name in ("BINUNICODE", "BININT", "BININT1", "BININT2", "BINFLOAT"):
            stack.append(arg)
        elif name == "NONE": stack.append(None)
        elif name == "NEWTRUE": stack.append(True)
        elif name == "NEWFALSE": stack.append(False)
        elif name in ("BINPUT", "LONG_BINPUT"):
            memo[arg] = stack[-1]
        elif name in ("BINGET", "LONG_BINGET"):
            stack.append(memo[arg])
        elif name == "GLOBAL":
            if arg not in GLOBALS:
                raise ValueError(f"Forbidden GLOBAL: {arg}")
            stack.append(GlobalName(arg))
        elif name == "TUPLE": stack.append(tuple(marked()))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            n = int(name[-1]); values = tuple(stack[-n:]); del stack[-n:]; stack.append(values)
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop()
            if not isinstance(stack[-1], dict) or not isinstance(key, (str, int)):
                raise ValueError("Malformed SETITEM")
            stack[-1][key] = value
        elif name == "SETITEMS":
            pairs = marked()
            if not isinstance(stack[-1], dict) or len(pairs) % 2:
                raise ValueError("Malformed SETITEMS")
            for k, v in zip(pairs[::2], pairs[1::2]):
                if not isinstance(k, (str, int)):
                    raise ValueError("Unsupported mapping key")
                stack[-1][k] = v
        elif name == "NEWOBJ":
            args, cls = stack.pop(), stack.pop()
            if cls != GlobalName("argparse Namespace") or args != ():
                raise ValueError("Unsupported NEWOBJ")
            stack.append(StaticMap("namespace"))
        elif name == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], StaticMap) or not isinstance(state, dict):
                raise ValueError("Unsupported BUILD")
            stack[-1].attributes = state  # never call __setstate__
        elif name == "BINPERSID":
            desc = stack.pop()
            if not isinstance(desc, tuple) or len(desc) != 5 or desc[0] != "storage":
                raise ValueError("Unsupported storage description")
            _, kind, key, location, size = desc
            if kind not in tuple(GlobalName(x) for x in ("torch FloatStorage", "torch HalfStorage", "torch LongStorage")):
                raise ValueError("Unsupported storage type")
            if not isinstance(key, str) or not key.isdigit() or type(size) is not int or size < 0:
                raise ValueError("Malformed storage key/size")
            stack.append(StorageSpec(kind.name, key, size))
        elif name == "REDUCE":
            args, fn = stack.pop(), stack.pop()
            if fn == GlobalName("collections OrderedDict") and args == ():
                stack.append(StaticMap("ordered_dict"))
            elif fn == GlobalName("torch._utils _rebuild_tensor_v2"):
                if not isinstance(args, tuple) or len(args) != 6:
                    raise ValueError("Unsupported tensor rebuild args")
                storage, offset, shape, stride, requires_grad, hooks = args
                if not isinstance(storage, StorageSpec) or type(offset) is not int or offset < 0:
                    raise ValueError("Invalid tensor storage/offset")
                if not isinstance(shape, tuple) or not isinstance(stride, tuple) or len(shape) != len(stride):
                    raise ValueError("Invalid shape/stride")
                if not all(type(n) is int and n >= 0 for n in shape + stride):
                    raise ValueError("Invalid tensor dimensions")
                stack.append(TensorSpec(storage, offset, shape, stride))
            else:
                raise ValueError(f"Unsupported REDUCE: {fn}")
        elif name == "STOP":
            if len(stack) != 1 or not isinstance(stack[0], dict):
                raise ValueError("Malformed root")
            return stack[0]
        else:
            raise ValueError(f"Unsupported opcode: {name}")
    raise ValueError("Missing STOP")


def extract_dpt(checkpoint: Path) -> tuple[dict, dict]:
    checkpoint = checkpoint.resolve()
    if checkpoint != ROOT / "checkpoints/output/2views.pth":
        raise ValueError("Only the authorized 2views checkpoint path is supported")
    stat = checkpoint.stat()
    if (stat.st_size, stat.st_mtime_ns) != (BASE_SIZE, BASE_MTIME_NS):
        raise ValueError("Checkpoint changed since B-00 SHA256 receipt; re-review required")
    started = time.perf_counter()
    with zipfile.ZipFile(checkpoint) as archive:
        names = archive.namelist()
        records = [n for n in names if n.endswith("/data.pkl")]
        if len(records) != 1:
            raise ValueError("Ambiguous archive metadata")
        prefix = records[0][:-len("data.pkl")]
        byteorder = archive.read(prefix + "byteorder").decode() if prefix + "byteorder" in names else "little"
        if byteorder != "little" or sys.byteorder != "little":
            raise ValueError("Only little-endian tensor storage is supported")
        raw = archive.read(records[0])
        tree = parse_tensor_descriptions(raw)
        if not isinstance(tree.get("model"), dict):
            raise ValueError("No model state mapping")
        selected = {k[len("dpt_head."):]: v for k, v in tree["model"].items() if k.startswith("dpt_head.")}
        if not selected or not all(isinstance(v, TensorSpec) for v in selected.values()):
            raise ValueError("No valid dpt_head.* tensors")
        dtype_map = {"torch FloatStorage": torch.float32, "torch HalfStorage": torch.float16}
        state, storage_bytes = {}, 0
        for key, desc in selected.items():
            if desc.storage.kind not in dtype_map:
                raise ValueError(f"Non-floating DPT storage: {key}")
            dtype = dtype_map[desc.storage.kind]
            nbytes = desc.storage.numel * torch.empty((), dtype=dtype).element_size()
            storage_bytes += nbytes
            if nbytes > 256 * 1024**2 or storage_bytes > 512 * 1024**2:
                raise ValueError("DPT extraction memory bound exceeded")
            record = prefix + "data/" + desc.storage.key
            if archive.getinfo(record).file_size != nbytes:
                raise ValueError(f"Storage byte count mismatch: {key}")
            end = desc.offset + sum((n-1)*s for n, s in zip(desc.shape, desc.stride))
            if not desc.shape or min(desc.shape) <= 0 or end >= desc.storage.numel:
                raise ValueError(f"Invalid DPT storage range: {key}")
            # Read only the selected DPT records; archive CRC is checked by zipfile.
            buffer = bytearray(archive.read(record))
            tensor = torch.frombuffer(buffer, dtype=dtype)
            state[key] = tensor.as_strided(desc.shape, desc.stride, desc.offset).clone()
    reference = make_reference_dpt()
    checked = strict_load_dpt(reference, state)
    report = {"status": "DPT_INIT_VERIFIED", "source_checkpoint": str(checkpoint),
              "source_checkpoint_sha256": BASE_SHA256,
              "hash_provenance": "B-00 full hash receipt; B-01 size/mtime unchanged; no full rehash",
              "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
              "source_prefix": "model.dpt_head.", "source_dtype_counts": {},
              "method": "static protocol-2 opcode interpretation + selected ZIP storage reads; no pickle execution",
              "pickle_sha256": hashlib.sha256(raw).hexdigest(), "dpt_storage_bytes_read": storage_bytes,
              "strict_load": checked, "elapsed_seconds": time.perf_counter()-started,
              "purpose": "INITIALIZATION_ONLY_NOT_TRAINED_STUDENT",
              "keys_shapes": {k: list(v.shape) for k, v in state.items()}}
    for t in state.values():
        name = str(t.dtype)
        report["source_dtype_counts"][name] = report["source_dtype_counts"].get(name, 0) + 1
    return state, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "instance_B"):
        raise ValueError("Output must stay in instance_B")
    state, report = extract_dpt(ROOT / "checkpoints/output/2views.pth")
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "dpt_init_from_2views.pt"
    receipt = out / "dpt_init_receipt.json"
    if weights.exists() or receipt.exists():
        raise FileExistsError("Refusing to overwrite initialization artifacts")
    with weights.open("xb") as stream:
        torch.save(state, stream)
    report["artifact_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    report["artifact_bytes"] = weights.stat().st_size
    report["artifact_path"] = str(weights.relative_to(ROOT))
    # Verify the small pure-tensor artifact through the existing safe loader.
    loaded = torch.load(weights, map_location="cpu", weights_only=True)
    if set(loaded) != set(state) or any(not torch.equal(loaded[k], state[k]) for k in state):
        raise RuntimeError("Extracted artifact readback mismatch")
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k != "keys_shapes"}, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
