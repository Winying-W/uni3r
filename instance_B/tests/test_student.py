"""B-01 synthetic tests of the actual student and original pretrained DPT."""
from dataclasses import replace
import gc
import io
import json
import os
from pathlib import Path
import unittest

import torch

from instance_B.dpt_adapter import ROOT, PatchOnlyDPT, make_reference_dpt, strict_load_dpt
from instance_B.extract_dpt_init import parse_tensor_descriptions
from instance_B.student import (
    InstanceStudent, StudentConfig, flatten_membership_logits, load_dpt_initialization,
    mask_logits_from_features, normalize_predicted_xyz,
)
from instance_B.tests.test_contracts import synthetic_inputs, record_dtypes
from uni3r.instance.contracts import ContractError, TOKEN_LAYERS, validate_instance_prediction

RUN_ROOT = ROOT / "instance_B/results/B-01/20261008_cpu_01"
EVIDENCE = {}


def small_config(q=7):
    return StudentConfig(height=32, width=32, num_queries=q, decoder_layers=2,
                         feedforward_dim=512, profile="synthetic_small")


def synthetic_scene(config, b=1, seed=20261008):
    """All scene/frame/source metadata remain explicitly SYNTHETIC."""
    batch, _, _ = synthetic_inputs(config.contract_spec, q=1, b=b)
    generator = torch.Generator().manual_seed(seed)
    def normal(shape):
        return torch.randn(shape, generator=generator)
    batch["images"] = torch.rand(batch["images"].shape, generator=generator) * 2 - 1
    batch["patch_tokens"] = {k: normal(t.shape) for k,t in batch["patch_tokens"].items()}
    batch["means"] = normal(batch["means"].shape)
    for i in range(b):
        batch["means"][i] += i * 100
    batch["geometry_confidence"] = torch.rand(batch["geometry_confidence"].shape, generator=generator) * 100 + 1
    record_dtypes(batch)
    return batch


def load_init():
    return load_dpt_initialization(RUN_ROOT / "dpt_init_from_2views.pt", RUN_ROOT / "dpt_init_receipt.json")


class STUDENT_SYNTHETIC_TEST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7301)
        cls.initialization = load_init()
        cls.model = InstanceStudent(small_config(), dpt_init=cls.initialization).eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model, cls.initialization
        gc.collect()

    def test_a_dpt_init_complete_and_independent_storage(self):
        model = self.model
        self.assertEqual(len(model.instance_dpt.state_dict()), 60)
        self.assertTrue(all(p.requires_grad for p in model.instance_dpt.parameters()))
        for key, tensor in model.instance_dpt.state_dict().items():
            source = self.initialization.state[key]
            self.assertTrue(torch.equal(tensor, source), key)
            self.assertNotEqual(tensor.data_ptr(), source.data_ptr(), key)
        key, parameter = next(iter(model.instance_dpt.named_parameters()))
        source_before = self.initialization.state[key].clone()
        saved = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(1)
        self.assertTrue(torch.equal(source_before, self.initialization.state[key]))
        with torch.no_grad():
            parameter.copy_(saved)
        EVIDENCE["DPT_INIT"] = model.initialization

    def test_b_DPT_ADAPTER_SYNTHETIC_PARITY(self):
        torch.manual_seed(991)
        reference = make_reference_dpt().eval().requires_grad_(False)
        strict_load_dpt(reference, self.initialization.state)
        reference.requires_grad_(False)
        batch = synthetic_scene(small_config(), b=2)
        # Independently generated complete 24-layer SYNTHETIC list for the
        # reference call. Never represent a four-layer cache as real 24 layers.
        original24 = [torch.randn(2, 2, 9, 2048) for _ in range(24)]
        cached = {k: original24[k][:, :, 5:, :].clone() for k in TOKEN_LAYERS}
        with torch.inference_mode():
            original = reference(original24, batch["images"], patch_start_idx=5, frames_chunk_size=None)
            adapted = self.model.instance_dpt(cached, batch["images"])
        torch.testing.assert_close(adapted, original, rtol=1e-5, atol=1e-6)
        self.assertEqual(reference.intermediate_layer_idx, [4, 11, 17, 23])
        self.assertEqual(self.model.instance_dpt.intermediate_layer_idx, [0, 1, 2, 3])
        for key, tensor in reference.state_dict().items():
            self.assertTrue(torch.equal(tensor, self.initialization.state[key]))
            self.assertNotEqual(tensor.data_ptr(), self.model.instance_dpt.state_dict()[key].data_ptr())
        # Changing excluded camera/register tokens must not affect either call.
        for value in original24:
            value[:, :, :5] = 12345
        with torch.inference_mode():
            changed = reference(original24, batch["images"], patch_start_idx=5, frames_chunk_size=None)
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
        EVIDENCE["DPT_ADAPTER_SYNTHETIC_PARITY"] = {"status": "PASS", "max_abs_error": (original-adapted).abs().max().item(),
                                                   "rtol": 1e-5, "atol": 1e-6, "batch": 2, "image_size": [32,32],
                                                   "source": "24 independent synthetic layers, original DPT weights"}

    def test_c_shapes_dtype_finite_b2(self):
        batch = synthetic_scene(small_config(), b=2)
        with torch.inference_mode():
            prediction = self.model(batch)
        expected = {"mask_logits": (2,7,2,32,32), "objectness_logits": (2,7), "query_features": (2,7,256)}
        for key, shape in expected.items():
            self.assertEqual(tuple(prediction[key].shape), shape)
            self.assertEqual(prediction[key].dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(prediction[key]).all()))
        EVIDENCE["small_shapes"] = {k:list(v.shape) for k,v in prediction.items()}

    def test_d_variable_q(self):
        for q in (1, 11):
            cfg = small_config(q)
            model = InstanceStudent(cfg, dpt_init=self.initialization).eval()
            with torch.inference_mode():
                pred = model(synthetic_scene(cfg))
            self.assertEqual(tuple(pred["mask_logits"].shape), (1,q,2,32,32))
            validate_instance_prediction(pred, spec=cfg.contract_spec)
            del model
        EVIDENCE["variable_Q"] = [1,7,11]

    def test_e_scene_independence(self):
        batch = synthetic_scene(small_config(), b=2)
        with torch.inference_mode():
            before = self.model(batch)
            batch["images"][1].neg_()
            batch["means"][1].mul_(30).add_(123)
            batch["geometry_confidence"][1].mul_(100)
            for t in batch["patch_tokens"].values():
                t[1].mul_(3)
            after = self.model(batch)
        for key in before:
            torch.testing.assert_close(before[key][0], after[key][0], rtol=0, atol=0)
        self.assertFalse(torch.equal(before["mask_logits"][1], after["mask_logits"][1]))

    def test_f_backward_all_active_student_parameters(self):
        batch = synthetic_scene(small_config(), b=2)
        model = self.model.train()
        model.zero_grad(set_to_none=True)
        snapshots = {k:t.clone() for k,t in self.initialization.state.items()}
        input_tensors = [batch["images"], batch["means"], batch["geometry_confidence"], *batch["patch_tokens"].values()]
        prediction = model(batch)
        objective = prediction["mask_logits"].square().mean() + prediction["objectness_logits"].square().mean()
        objective.backward()
        by_module = {}
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            nonzero = bool((parameter.grad != 0).any())
            self.assertTrue(nonzero, name)
            module = name.split(".")[0]
            by_module.setdefault(module, {"parameters_with_grad": 0, "finite_nonzero": 0})
            by_module[module]["parameters_with_grad"] += 1
            by_module[module]["finite_nonzero"] += int(nonzero)
        for tensor in input_tensors:
            self.assertIsNone(tensor.grad)
            self.assertFalse(tensor.requires_grad)
        for key,tensor in self.initialization.state.items():
            self.assertTrue(torch.equal(tensor, snapshots[key]), key)
        model.zero_grad(set_to_none=True)
        model.eval()
        EVIDENCE["gradient_check"] = {"objective_kind": "DIAGNOSTIC_LOGITS_SQUARE_NOT_INSTANCE_LOSS",
                                     "objective": objective.item(), "modules": by_module,
                                     "input_grads": "NONE", "reference_state_unchanged": True}

    def test_g_detach_in_adapter_and_reject_attached_contract(self):
        batch = synthetic_scene(small_config())
        batch["patch_tokens"][4].requires_grad_()
        with self.assertRaisesRegex(ContractError, "must not require gradients"):
            self.model(batch)
        # Independent adapter API explicitly detaches, even if misused outside
        # the shared contract; the full student does not silently repair a batch.
        self.model.zero_grad(set_to_none=True)
        self.model.instance_dpt(batch["patch_tokens"], batch["images"]).square().mean().backward()
        self.assertIsNone(batch["patch_tokens"][4].grad)
        self.model.zero_grad(set_to_none=True)

    def test_h_identical_xyz_confidence_scores(self):
        batch = synthetic_scene(small_config())
        batch["means"].fill_(42)
        batch["geometry_confidence"].fill_(10000)
        xyz, center, scale = normalize_predicted_xyz(batch["means"], 1e-6)
        self.assertEqual(xyz.abs().max().item(), 0)
        torch.testing.assert_close(center, torch.full_like(center,42), rtol=0,atol=0)
        self.assertEqual(scale.item(), 1e-6)
        with torch.inference_mode():
            pred = self.model(batch)
        self.assertTrue(bool(torch.isfinite(pred["mask_logits"]).all()))
        batch["geometry_confidence"].fill_(0)
        with torch.inference_mode():
            zero = self.model(batch)
        self.assertFalse(torch.equal(pred["mask_logits"], zero["mask_logits"]))

    def test_i_xyz_normalization_near_zero_and_no_batch_mix(self):
        means = torch.zeros(2,2,2,2,3,requires_grad=True)
        with torch.no_grad():
            means[0,0,0,0,0] = 1e-9
            means[1].fill_(1e30)
        normalized, center, scale = normalize_predicted_xyz(means, 1e-6)
        self.assertFalse(normalized.requires_grad)
        self.assertTrue(bool(torch.isfinite(normalized).all()))
        self.assertLess(normalized[0].abs().max().item(), 0.01)
        self.assertEqual(normalized[1].abs().max().item(), 0)

    def test_j_hand_checked_flatten_and_filter(self):
        logits = torch.arange(24,dtype=torch.float32).reshape(2,1,2,2,3)
        flat = flatten_membership_logits(logits)
        self.assertEqual(flat[0,0].tolist(), list(range(12)))
        self.assertEqual(flat[0,0,1*2*3+1*3+2].item(), 11)
        selected = flatten_membership_logits(logits,[torch.tensor([11,0,7]),torch.empty(0,dtype=torch.long)])
        self.assertEqual(selected[0].tolist(), [[11,0,7]])
        self.assertEqual(tuple(selected[1].shape),(1,0))
        for bad in (torch.tensor([12]),torch.tensor([1,1]),torch.tensor([1.0])):
            with self.assertRaises(ValueError):
                flatten_membership_logits(logits,[bad,torch.tensor([0])])

    def test_k_mask_dot_scale_raw_logits(self):
        query = torch.ones(1,2,64)
        query[:,1].neg_()
        pixels = torch.ones(1,2,64,2,2)
        logits = mask_logits_from_features(query,pixels)
        self.assertTrue(bool((logits[:,0] == 8).all()))
        self.assertTrue(bool((logits[:,1] == -8).all()))

    def test_l_eval_state_dict_save_reload(self):
        batch = synthetic_scene(small_config())
        model = self.model.eval()
        with torch.inference_mode():
            before = model(batch)
        location = os.environ.get("B01_TEST_ARTIFACT_DIR")
        if location:
            path = Path(location).resolve()/"eval_roundtrip_state_dict.pt"
            self.assertTrue(path.is_relative_to(ROOT/"instance_B"))
            with path.open("xb") as stream:
                torch.save(model.state_dict(),stream)
            state = torch.load(path,map_location="cpu",weights_only=True)
            save_kind = str(path.relative_to(ROOT))
        else:
            stream=io.BytesIO(); torch.save(model.state_dict(),stream); stream.seek(0)
            state=torch.load(stream,map_location="cpu",weights_only=True)
            save_kind="in_memory_roundtrip"
        restored = InstanceStudent(small_config(),dpt_init=self.initialization).eval()
        restored.load_state_dict(state,strict=True)
        with torch.inference_mode():
            after=restored(batch)
        for key in before:
            torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
        EVIDENCE["state_dict_roundtrip"]={"status":"PASS","max_abs_error":0.0,"artifact":save_kind}

    def test_m_missing_initialization_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"DPT_INIT_BLOCKED"):
            InstanceStudent(StudentConfig())
        cfg=replace(small_config(),init_mode="synthetic_random_init")
        model=InstanceStudent(cfg).eval()
        with torch.inference_mode():
            model(synthetic_scene(cfg))
        self.assertEqual(model.initialization["mode"],"synthetic_random_init")

    def test_n_reject_missing_or_wrong_dpt_weights(self):
        reference=self.model.instance_dpt
        bad=dict(self.initialization.state)
        bad.pop(next(iter(bad)))
        with self.assertRaisesRegex(ValueError,"keys mismatch"):
            strict_load_dpt(reference,bad)
        bad=dict(self.initialization.state)
        key=next(iter(bad)); bad[key]=bad[key][:1]
        with self.assertRaisesRegex(ValueError,"shape mismatch"):
            strict_load_dpt(reference,bad)

    def test_o_tokens_order_extra_special_and_rgb(self):
        batch=synthetic_scene(small_config())
        normal=batch["patch_tokens"]
        reverse={k:normal[k] for k in reversed(TOKEN_LAYERS)}
        with torch.inference_mode():
            a=self.model.instance_dpt(normal,batch["images"])
            b=self.model.instance_dpt(reverse,batch["images"])
            dark=self.model.instance_dpt(normal,torch.zeros_like(batch["images"]))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(torch.equal(a,dark))
        reverse[4]=torch.zeros(1,2,5,2048)
        with self.assertRaisesRegex(ValueError,"patch-only"):
            self.model.instance_dpt(reverse,batch["images"])
        batch["metadata"]["rgb_normalization"]="zero_to_one"
        batch["images"].abs_()
        with self.assertRaisesRegex(ValueError,"RGB metadata"):
            self.model(batch)

    def test_p_default_config_and_memory_bound(self):
        config=StudentConfig.from_json(ROOT/"instance_B/configs/student_v0.json")
        self.assertEqual(config,StudentConfig())
        self.assertEqual(config.memory_token_count,2560)
        self.assertEqual(small_config().memory_token_count,40)
        with self.assertRaisesRegex(ValueError,"8192"):
            replace(config,memory_strides=(1,))

    def test_q_no_gt_teacher_dependency(self):
        batch=synthetic_scene(small_config())
        with torch.inference_mode():
            before=self.model(batch)
            batch.update(gt_depth="FORBIDDEN_SENTINEL",gt_extrinsics=object(),instance_targets=object(),teacher=object())
            after=self.model(batch)
        for key in before:
            torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)

    def test_r_static_parser_rejects_executable_global(self):
        with self.assertRaisesRegex(ValueError,"Forbidden GLOBAL"):
            parse_tensor_descriptions(b"\x80\x02cos\nsystem\n.")

    def test_s_attention_operates_only_on_queries(self):
        observed=[]
        handles=[]
        for layer in self.model.decoder:
            handles.append(layer.self_attention.register_forward_pre_hook(lambda m,args: observed.append(("self",args[0].shape[1],args[1].shape[1]))))
            handles.append(layer.cross_attention.register_forward_pre_hook(lambda m,args: observed.append(("cross",args[0].shape[1],args[1].shape[1]))))
        try:
            with torch.inference_mode(): self.model(synthetic_scene(small_config()))
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(observed,[("self",7,7),("cross",7,40)]*2)
        EVIDENCE["attention_shapes"]=observed


if __name__=="__main__":
    unittest.main(verbosity=2)
