"""Exercise the unmodified author's DSLR preprocessing on a known test plane.

This verifies environment/geometry/units only. It is not ScanNet++ data and
does not establish the dataset release or training-scene membership.
"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from munch import Munch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_process/scannetpp"))
from common.render import process_scene


def main():
    fixture = ROOT / ".cache/author_render_smoke"
    scene_id = "synthetic_plane_only"
    scene = fixture / "raw/data" / scene_id
    for folder in ("scans", "dslr/colmap", "dslr/resized_images", "dslr/resized_anon_masks"):
        (scene / folder).mkdir(parents=True, exist_ok=True)
    # A two-sided, coloured plane at world z=2 metres.
    mesh = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face 4
property list uchar int vertex_indices
end_header
-2 -2 2 255 128 0
2 -2 2 255 128 0
2 2 2 255 128 0
-2 2 2 255 128 0
3 0 1 2
3 0 2 3
3 2 1 0
3 3 2 0
"""
    (scene / "scans/mesh_aligned_0.05.ply").write_text(mesh)
    (scene / "dslr/colmap/cameras.txt").write_text(
        "1 OPENCV_FISHEYE 192 144 140 140 96 72 0 0 0 0\n"
    )
    (scene / "dslr/colmap/images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 DSC00001.JPG\n\n"
        "2 1 0 0 0 0 0 -0.25 1 DSC00002.JPG\n\n"
    )
    (scene / "dslr/colmap/points3D.txt").write_text("")
    for name in ("DSC00001", "DSC00002"):
        rgb = np.full((144, 192, 3), (32, 96, 192), dtype=np.uint8)
        mask = np.full((144, 192), 255, dtype=np.uint8)
        assert cv2.imwrite(str(scene / f"dslr/resized_images/{name}.JPG"), rgb)
        assert cv2.imwrite(str(scene / f"dslr/resized_anon_masks/{name}.png"), mask)
    cfg = Munch(
        data_root=str(fixture / "raw"), output_dir=str(fixture / "processed"),
        render_dslr=True, render_iphone=False, downscale_factor=3,
        near=0.05, far=20.0,
    )
    process_scene(scene_id, 0, cfg)
    out = fixture / "processed" / scene_id / "dslr"
    frames = []
    for name, camera_z, expected_mm in (("DSC00001", 0.0, 2000), ("DSC00002", 0.25, 1750)):
        depth = cv2.imread(str(out / f"render_depth/{name}.png"), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(out / f"rgb_resized_undistorted/{name}.JPG"))
        mask = cv2.imread(str(out / f"mask_resized_undistorted/{name}.png"), cv2.IMREAD_UNCHANGED)
        camera = np.load(out / f"camera/{name}.npz")
        assert depth.shape == (48, 64) and depth.dtype == np.uint16
        assert rgb.shape == (48, 64, 3) and mask.shape == (48, 64)
        assert np.all(mask == 255)
        assert np.count_nonzero(depth) == depth.size
        max_error = int(np.abs(depth.astype(np.int32) - expected_mm).max())
        assert max_error <= 1, (name, max_error)
        expected_pose = np.eye(4, dtype=np.float32)
        expected_pose[2, 3] = camera_z
        np.testing.assert_allclose(camera["extrinsic"], expected_pose, atol=1e-6)
        intrinsic = camera["intrinsic"]
        assert intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all()
        np.testing.assert_allclose(intrinsic[:2, 2], (32, 24), atol=1e-6)
        frames.append({"frame": name, "expected_depth_mm": expected_mm,
                       "median_depth_mm": float(np.median(depth)), "max_depth_error_mm": max_error,
                       "camera_z_metres": camera_z, "output_shape": list(depth.shape)})
    import renderpy
    report = {"status": "passed", "test_data": "synthetic; not ScanNet++",
              "author_preprocessor": "data_process/scannetpp/common/render.py::process_scene",
              "dataset_version_verified": False, "python": sys.version,
              "renderpy_module": renderpy.__file__, "opencv_python": cv2.__version__,
              "numpy": np.__version__, "frames": frames}
    target = ROOT / "test_results/author_renderer_smoke.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
