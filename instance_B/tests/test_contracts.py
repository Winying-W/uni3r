"""CONTRACT_SYNTHETIC_TEST only. No generated tensor is a Uni3R fixture.

Run from repository root with `python -B -m unittest discover -s instance_B/tests -v`.
Small tests explicitly select a 16x16 synthetic profile; the default-profile test
also checks the team's exact 256x256 / 256 patches / 2048 channels / Q=100 shapes.
"""
import copy
import unittest

import torch

from uni3r.instance.contracts import (
    CONTRACT_STATUS, CONTRACT_VERSION, DEFAULT_SPEC, GAUSSIAN_FLATTEN_ORDER,
    TOKEN_LAYERS, ContractError, ContractSpec, scene_instance_ids,
    validate_frozen_scene_batch, validate_instance_prediction,
    validate_instance_targets, validate_training_batch,
)

SMALL = ContractSpec(height=16, width=16)


def record_dtypes(batch):
    entries = {}
    def visit(value, path):
        if isinstance(value, torch.Tensor):
            entries[path] = str(value.dtype)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(item, f"{path}[{i}]")
    visit({k: v for k, v in batch.items() if k != "metadata"}, "")
    batch["metadata"]["tensor_dtypes"] = entries


def synthetic_inputs(spec=SMALL, q=3, b=1):
    v, h, w = spec.views, spec.height, spec.width
    ids = torch.zeros((b, v, h, w), dtype=torch.long)
    ids[:, 0, 0, 0] = 7
    ids[:, 1, 0, 1] = 7
    ids[:, 1, 1, 1] = 29  # visible in only one view
    ids[:, 0, -1, -1] = -1
    valid = ids != -1
    scenes = [f"SYNTHETIC_scene_{i}" for i in range(b)]
    frames = [["SYNTHETIC_frame_0", "SYNTHETIC_frame_1"] for _ in range(b)]
    crop = {"original_size": [h, w], "crop_xyxy": [0, 0, w, h],
            "resize_hw": [h, w], "description": "SYNTHETIC identity crop/resize; no resampling"}
    batch = {
        "images": torch.zeros(b, v, 3, h, w),
        "intrinsics": torch.eye(3).repeat(b, v, 1, 1),
        "patch_tokens": {layer: torch.zeros(b, v, (h//spec.patch_size)*(w//spec.patch_size), spec.token_dim)
                         for layer in TOKEN_LAYERS},
        "means": torch.zeros(b, v, h, w, 3),
        "geometry_confidence": torch.full((b, v, h, w), 2.0),
        "gaussians": {},  # no fabricated model payload
        "metadata": {
            "contract_version": CONTRACT_VERSION, "evidence_kind": "synthetic",
            "base_checkpoint": {"name": "SYNTHETIC_NO_CHECKPOINT", "sha256": None},
            "uni3r_git_commit": "SYNTHETIC", "uni3r_worktree_dirty": False,
            "input_size": [h, w], "rgb_normalization": "minus_one_to_one",
            "token_layers": list(TOKEN_LAYERS), "patch_token_order": "row_major_yx",
            "special_tokens_removed": True,
            "special_token_removal": "SYNTHETIC tensors have no special tokens",
            "scene_id": scenes, "frame_ids": frames,
            "crop_resize": [[copy.deepcopy(crop) for _ in range(v)] for _ in range(b)],
            "intrinsics_convention": "pixel", "coordinate_system": "SYNTHETIC; no physical frame",
            "means_source": "synthetic", "geometry_confidence_semantics": "SYNTHETIC nonnegative score",
            "gaussian_flatten_order": GAUSSIAN_FLATTEN_ORDER,
            "gaussian_filtered": False, "gaussian_schema": "SYNTHETIC empty opaque payload",
        },
    }
    record_dtypes(batch)
    targets = {"instance_ids": ids, "valid_mask": valid, "confidence": valid.float(),
               "scene_id": copy.deepcopy(scenes), "frame_ids": copy.deepcopy(frames), "label_source": "gt"}
    prediction = {"mask_logits": torch.full((b, q, v, h, w), -2.0),
                  "objectness_logits": torch.full((b, q), 3.0),
                  "query_features": torch.zeros(b, q, spec.query_dim)}
    return batch, targets, prediction


class CONTRACT_SYNTHETIC_TEST(unittest.TestCase):
    def setUp(self):
        self.batch, self.targets, self.prediction = synthetic_inputs()

    def test_version_and_status(self):
        self.assertEqual(CONTRACT_VERSION, "0.1.0")
        self.assertEqual(CONTRACT_STATUS, "PROPOSED_FOR_TEAM_INTEGRATION")

    def test_valid_small_batch(self):
        validate_training_batch(self.batch, self.targets, self.prediction, spec=SMALL)

    def test_default_target_dimensions_q100(self):
        batch, targets, prediction = synthetic_inputs(DEFAULT_SPEC, q=100)
        validate_training_batch(batch, targets, prediction)
        self.assertEqual(tuple(batch["patch_tokens"][4].shape), (1, 2, 256, 2048))

    def test_input_shapes_rejected(self):
        for field in ("images", "intrinsics", "means", "geometry_confidence"):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.batch)
                bad[field] = bad[field][..., :-1]
                with self.assertRaisesRegex(ContractError, field):
                    validate_frozen_scene_batch(bad, spec=SMALL)

    def test_special_token_shape_and_layer_keys(self):
        self.batch["patch_tokens"][4] = torch.zeros(1, 2, 2, 2048)
        with self.assertRaisesRegex(ContractError, "patch_tokens.4.*shape"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)
        self.batch["patch_tokens"][5] = self.batch["patch_tokens"].pop(4)
        with self.assertRaisesRegex(ContractError, "layer keys"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_required_dtypes(self):
        for field, dtype in (("instance_ids", torch.int32), ("valid_mask", torch.uint8), ("confidence", torch.long)):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.targets)
                bad[field] = bad[field].to(dtype)
                with self.assertRaisesRegex(ContractError, field + ".*dtype"):
                    validate_instance_targets(bad, spec=SMALL)
        self.batch["images"] = self.batch["images"].to(torch.uint8)
        with self.assertRaisesRegex(ContractError, "images.*dtype"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_nan_inf_rejected(self):
        for field in ("images", "intrinsics", "means", "geometry_confidence"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    bad = copy.deepcopy(self.batch)
                    bad[field].reshape(-1)[0] = value
                    with self.assertRaisesRegex(ContractError, "NaN/Inf"):
                        validate_frozen_scene_batch(bad, spec=SMALL)
        self.batch["patch_tokens"][11][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ContractError, "NaN/Inf"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_ignored_nan_not_hidden(self):
        self.targets["confidence"][0, 0, -1, -1] = float("nan")
        with self.assertRaisesRegex(ContractError, "NaN/Inf"):
            validate_instance_targets(self.targets, spec=SMALL)

    def test_id_rules_background_and_unknown(self):
        validate_instance_targets(self.targets, spec=SMALL)
        self.assertEqual(scene_instance_ids(self.targets, spec=SMALL)[0].tolist(), [7, 29])
        self.targets["instance_ids"][0, 0, 0, 0] = -2
        with self.assertRaisesRegex(ContractError, "only -1, 0 or positive"):
            validate_instance_targets(self.targets, spec=SMALL)

    def test_unknown_cannot_be_valid(self):
        self.targets["valid_mask"][0, 0, -1, -1] = True
        with self.assertRaisesRegex(ContractError, "unknown ID -1"):
            validate_instance_targets(self.targets, spec=SMALL)

    def test_invalid_pixel_must_have_zero_weight(self):
        self.targets["confidence"][0, 0, -1, -1] = 0.25
        with self.assertRaisesRegex(ContractError, "zero weight"):
            validate_instance_targets(self.targets, spec=SMALL)

    def test_confidence_range_and_fractional_weights(self):
        self.targets["label_source"] = "pseudo"
        self.targets["confidence"] *= 0.25
        validate_instance_targets(self.targets, spec=SMALL)
        for value in (-0.01, 1.01, float("inf")):
            with self.subTest(value=value):
                self.targets["confidence"][0, 0, 0, 0] = value
                with self.assertRaises(ContractError):
                    validate_instance_targets(self.targets, spec=SMALL)

    def test_scene_union_and_single_view_visibility(self):
        ids = scene_instance_ids(self.targets, spec=SMALL)[0]
        self.assertEqual(ids.tolist(), [7, 29])
        self.targets["instance_ids"][0, 0, 0, 0] = 0
        self.assertEqual(scene_instance_ids(self.targets, spec=SMALL)[0].tolist(), [7, 29])

    def test_scene_local_ids_not_batch_global(self):
        _, targets, _ = synthetic_inputs(b=2)
        targets["instance_ids"][1].zero_()
        result = scene_instance_ids(targets, spec=SMALL)
        self.assertEqual(result[0].tolist(), [7, 29])
        self.assertEqual(result[1].tolist(), [])

    def test_empty_background_and_all_ignore_scenes(self):
        for label in (0, -1):
            with self.subTest(label=label):
                self.targets["instance_ids"].fill_(label)
                self.targets["valid_mask"].fill_(label == 0)
                self.targets["confidence"].fill_(1 if label == 0 else 0)
                validate_instance_targets(self.targets, spec=SMALL)
                self.assertEqual(scene_instance_ids(self.targets, spec=SMALL)[0].numel(), 0)

    def test_invalid_positive_and_zero_weight_ids_omitted(self):
        self.targets["valid_mask"].zero_()
        self.targets["confidence"].zero_()
        self.assertEqual(scene_instance_ids(self.targets, spec=SMALL)[0].numel(), 0)

    def test_prediction_shapes_and_flexible_query_count(self):
        for q in (1, 7, 100):
            _, _, pred = synthetic_inputs(q=q)
            validate_instance_prediction(pred, spec=SMALL)
        for field in self.prediction:
            with self.subTest(field=field):
                pred = copy.deepcopy(self.prediction)
                pred[field] = pred[field][..., :-1]
                with self.assertRaisesRegex(ContractError, field):
                    validate_instance_prediction(pred, spec=SMALL)

    def test_logits_allow_negative_and_greater_than_one_and_grad(self):
        self.prediction["mask_logits"].requires_grad_()
        validate_instance_prediction(self.prediction, spec=SMALL)
        self.assertTrue(self.prediction["mask_logits"].requires_grad)

    def test_prediction_nan_and_integer_dtype(self):
        for field in self.prediction:
            for value in (float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    pred = copy.deepcopy(self.prediction)
                    pred[field].reshape(-1)[0] = value
                    with self.assertRaisesRegex(ContractError, "NaN/Inf"):
                        validate_instance_prediction(pred, spec=SMALL)
        self.prediction["mask_logits"] = self.prediction["mask_logits"].long()
        with self.assertRaisesRegex(ContractError, "dtype"):
            validate_instance_prediction(self.prediction, spec=SMALL)

    def test_missing_metadata_precise_errors(self):
        for key in list(self.batch["metadata"]):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.batch)
                del bad["metadata"][key]
                with self.assertRaisesRegex(ContractError, rf"metadata\.{key}: missing"):
                    validate_frozen_scene_batch(bad, spec=SMALL)

    def test_metadata_dtype_and_special_tokens(self):
        self.batch["metadata"]["tensor_dtypes"]["means"] = "torch.float64"
        with self.assertRaisesRegex(ContractError, "tensor_dtypes.means"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)
        record_dtypes(self.batch)
        self.batch["metadata"]["special_tokens_removed"] = False
        with self.assertRaisesRegex(ContractError, "special_tokens_removed"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_metadata_crop_and_normalization(self):
        self.batch["metadata"]["crop_resize"][0][0]["crop_xyxy"] = [0, 0, 17, 16]
        with self.assertRaisesRegex(ContractError, "crop outside"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)
        self.batch["metadata"]["crop_resize"][0][0]["crop_xyxy"] = [0, 0, 16, 16]
        self.batch["images"][0, 0, 0, 0, 0] = 2
        with self.assertRaisesRegex(ContractError, "rgb_normalization"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_detached_inputs_required(self):
        for field in ("images", "means", "geometry_confidence"):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.batch)
                bad[field].requires_grad_()
                with self.assertRaisesRegex(ContractError, "must not require gradients"):
                    validate_frozen_scene_batch(bad, spec=SMALL)
        self.batch["patch_tokens"][23].requires_grad_()
        with self.assertRaisesRegex(ContractError, "must not require gradients"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_geometry_confidence_is_not_label_probability(self):
        validate_frozen_scene_batch(self.batch, spec=SMALL)  # score=2 is legal
        self.batch["geometry_confidence"][0, 0, 0, 0] = -1
        with self.assertRaisesRegex(ContractError, "nonnegative"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_gaussian_filter_mapping(self):
        self.batch["metadata"]["gaussian_filtered"] = True
        with self.assertRaisesRegex(ContractError, "required after filtering"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)
        self.batch["gaussians"]["source_indices"] = [torch.tensor([0, 16*16+2*16+3])]
        record_dtypes(self.batch)
        validate_frozen_scene_batch(self.batch, spec=SMALL)
        self.batch["gaussians"]["source_indices"][0][1] = 512
        with self.assertRaisesRegex(ContractError, "out of range"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_gaussian_filter_duplicates_dtype_and_empty(self):
        self.batch["metadata"]["gaussian_filtered"] = True
        for vector, message in ((torch.tensor([0, 0]), "duplicate"), (torch.tensor([0.0]), "dtype")):
            self.batch["gaussians"]["source_indices"] = [vector]
            record_dtypes(self.batch)
            with self.assertRaisesRegex(ContractError, message):
                validate_frozen_scene_batch(self.batch, spec=SMALL)
        self.batch["gaussians"]["source_indices"] = [torch.empty(0, dtype=torch.long)]
        record_dtypes(self.batch)
        validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_opaque_gaussian_payload_finiteness(self):
        self.batch["gaussians"]["synthetic_test_value"] = torch.tensor([float("nan")])
        record_dtypes(self.batch)
        with self.assertRaisesRegex(ContractError, "NaN/Inf"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_frame_scene_and_batch_alignment(self):
        self.targets["frame_ids"][0].reverse()
        with self.assertRaisesRegex(ContractError, "frame_ids"):
            validate_training_batch(self.batch, self.targets, spec=SMALL)
        self.targets["frame_ids"][0].reverse()
        self.targets["scene_id"][0] = "different_scene"
        with self.assertRaisesRegex(ContractError, "scene_id"):
            validate_training_batch(self.batch, self.targets, spec=SMALL)
        _, target2, _ = synthetic_inputs(b=2)
        with self.assertRaisesRegex(ContractError, "batch size"):
            validate_training_batch(self.batch, target2, spec=SMALL)

    def test_no_input_mutation(self):
        tensors = [self.batch["images"], self.batch["means"], *self.batch["patch_tokens"].values(),
                   self.targets["instance_ids"], self.targets["valid_mask"], self.targets["confidence"],
                   *self.prediction.values()]
        before = [t.clone() for t in tensors]
        pointers = [t.data_ptr() for t in tensors]
        meta = copy.deepcopy(self.batch["metadata"])
        validate_training_batch(self.batch, self.targets, self.prediction, spec=SMALL)
        scene_instance_ids(self.targets, spec=SMALL)
        for original, snapshot, pointer in zip(tensors, before, pointers):
            self.assertTrue(torch.equal(original, snapshot))
            self.assertEqual(original.data_ptr(), pointer)
        self.assertEqual(self.batch["metadata"], meta)

    def test_synthetic_cannot_claim_real_checkpoint(self):
        self.batch["metadata"]["base_checkpoint"] = {"name": "2views.pth", "sha256": "0"*64}
        with self.assertRaisesRegex(ContractError, "must not claim checkpoint"):
            validate_frozen_scene_batch(self.batch, spec=SMALL)

    def test_real_provenance_validation_negative_only(self):
        # Synthetic tensors deliberately carry invalid real declarations here.
        # Rejection is a contract unit test, never evidence of real integration.
        batch, _, _ = synthetic_inputs(DEFAULT_SPEC)
        batch["metadata"]["evidence_kind"] = "real"
        with self.assertRaisesRegex(ContractError, "requires 2views.pth"):
            validate_frozen_scene_batch(batch)
        batch["metadata"]["base_checkpoint"] = {"name": "2views.pth", "sha256": "invalid"}
        with self.assertRaisesRegex(ContractError, "SHA256"):
            validate_frozen_scene_batch(batch)
        batch["metadata"]["base_checkpoint"]["sha256"] = "0"*64
        batch["metadata"]["uni3r_git_commit"] = "0"*40
        batch["metadata"]["means_source"] = "gt_xyz"
        with self.assertRaisesRegex(ContractError, "GT XYZ is forbidden"):
            validate_frozen_scene_batch(batch)


class REAL_UNI3R_FIXTURE_TEST(unittest.TestCase):
    @unittest.skip("NOT RUN: A has not supplied a real feature fixture inside the authorized root; replace this placeholder when integration is authorized")
    def test_real_fixture_not_available(self):
        pass


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
