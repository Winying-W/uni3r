"""CPU verification for a delivered A fixture; no base model/checkpoint needed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from uni3r.instance.contracts import validate_frozen_scene_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text())
    assert report["status"] == "A_M0_REAL_FIXTURE_VERIFIED"
    for name, checksum in report["artifact_sha256"].items():
        h = hashlib.sha256()
        with (args.directory / name).open("rb") as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b""):
                h.update(chunk)
        assert h.hexdigest() == checksum, f"Changed artifact: {name}"
    batch = torch.load(args.directory / "fixture.pt", map_location="cpu", weights_only=True)
    validate_frozen_scene_batch(batch)
    assert batch["metadata"]["purpose"] == "M0_DIAGNOSTIC_ONLY_NOT_FORMAL_TRAIN_OR_VAL"
    assert not batch["metadata"]["gaussian_filtered"]
    views = batch["gaussians"]["views"]
    assert len(views) == 2
    attributes = dict(means=(3,), scales=(3,), rotations=(4,), covs=(3, 3),
                      opacities=(1,), sh_coeffs=(16, 3), gs_feats=(64,))
    b = batch["images"].shape[0]
    for view in views:
        for key, tail in attributes.items():
            assert view[key].shape == (b, 256, 256, *tail), key
    assert torch.equal(torch.stack([v["means"] for v in views], 1), batch["means"])
    for source in batch["gaussians"]["source_indices"]:
        assert torch.equal(source, torch.arange(2*256*256))
    assert json.loads((args.directory / "metadata.json").read_text()) == batch["metadata"]
    print("PASS: hashes, contract, Gaussian schema, source-pixel order, metadata and diagnostic-only scope")


if __name__ == "__main__":
    main()
