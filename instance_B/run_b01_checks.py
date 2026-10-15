"""Reproducible, CPU-only B-01 checks; no instance loss or formal training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest

import torch

from instance_B.dpt_adapter import ROOT
from instance_B.student import InstanceStudent, StudentConfig
from instance_B.tests import test_contracts, test_student


def write_json(path, value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2)
        stream.write("\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--diagnostic-steps",type=int,default=0,choices=[0,1,2,3])
    args=parser.parse_args()
    out=args.run_dir.resolve()
    if not out.is_relative_to(ROOT/"instance_B/results/B-01"):
        raise ValueError("Run directory must remain in B results")
    out.mkdir(parents=True,exist_ok=False)
    os.environ["B01_TEST_ARTIFACT_DIR"]=str(out)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("Run with CUDA_VISIBLE_DEVICES='' to protect shared GPUs")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7301)
    env={"command":" ".join(sys.argv),"python":sys.executable,"torch":torch.__version__,
         "device":"cpu","dtype":"float32","threads":torch.get_num_threads(),
         "interop_threads":torch.get_num_interop_threads(),"evidence_kind":"synthetic",
         "REAL_INTEGRATION":"NOT RUN","small_config":test_student.small_config().to_dict(),
         "full_config":StudentConfig().to_dict()}
    write_json(out/"configuration_and_command.json",env)
    start=time.perf_counter()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(test_contracts),
                             unittest.defaultTestLoader.loadTestsFromModule(test_student)])
    with (out/"unittest_output.txt").open("x",encoding="utf-8") as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print((out/"unittest_output.txt").read_text(),flush=True)
    summary={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
             "skipped":[{"test":str(t),"reason":reason} for t,reason in result.skipped],
             "unit_seconds":time.perf_counter()-start,"evidence":test_student.EVIDENCE}
    write_json(out/"unit_results.json",summary)
    if not result.wasSuccessful():
        print("CHECKS_FAILED: retained full tracebacks; full config and optimizer diagnostics NOT RUN",flush=True)
        return 1
    gc.collect()
    init=test_student.load_init()
    config=StudentConfig.from_json(ROOT/"instance_B/configs/student_v0.json")
    full=InstanceStudent(config,dpt_init=init).eval()
    scene=test_student.synthetic_scene(config)
    started=time.perf_counter()
    with torch.inference_mode():
        prediction=full(scene)
    full_result={"status":"PASS","evidence_kind":"synthetic","seconds":time.perf_counter()-started,
                 "parameter_count":sum(p.numel() for p in full.parameters()),
                 "memory_tokens":config.memory_token_count,"shapes":{k:list(v.shape) for k,v in prediction.items()},
                 "finite":{k:bool(torch.isfinite(v).all()) for k,v in prediction.items()},
                 "process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out/"full_config_forward.json",full_result)
    print("FULL_CONFIG_FORWARD",json.dumps(full_result),flush=True)
    del full,scene,prediction
    gc.collect()
    if args.diagnostic_steps:
        # Persistent admission marker prevents accidentally repeating steps when
        # rerunning checks. At most 3 small optimizer steps for all B-01 runs.
        marker=out.parent/"optimizer_diagnostic_admission.json"
        write_json(marker,{"run":str(out),"authorized_max_steps":args.diagnostic_steps,
                           "kind":"DIAGNOSTIC_NOT_INSTANCE_SUPERVISION"})
        torch.manual_seed(7301)
        small=InstanceStudent(test_student.small_config(),dpt_init=init).train()
        batch=test_student.synthetic_scene(small.config,b=2)
        optimizer=torch.optim.SGD(small.parameters(),lr=1e-4)
        original={k:v.clone() for k,v in init.state.items()}
        tracked={name:p.detach().clone() for name,p in small.named_parameters() if name in (
            "instance_dpt.projects.0.weight","object_queries","adapter.projections.0.weight",
            "decoder.0.cross_attention.in_proj_weight","mask_features.weight","objectness_head.weight")}
        records=[]
        started=time.perf_counter()
        for step in range(args.diagnostic_steps):
            optimizer.zero_grad(set_to_none=True)
            pred=small(batch)
            scalar=pred["mask_logits"].square().mean()+pred["objectness_logits"].square().mean()
            scalar.backward()
            if not bool(torch.isfinite(scalar)) or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in small.parameters()):
                raise RuntimeError("Nonfinite diagnostic or missing student gradient")
            optimizer.step()
            records.append({"step":step+1,"diagnostic_scalar":scalar.item()})
            write_json(out/f"diagnostic_step_{step+1}.json",records[-1])
        changes={name:(dict(small.named_parameters())[name].detach()-before).abs().max().item() for name,before in tracked.items()}
        if not all(delta>0 for delta in changes.values()):
            raise RuntimeError("Tracked student modules did not update")
        reference_unchanged=all(torch.equal(init.state[k],v) for k,v in original.items())
        if not reference_unchanged:
            raise RuntimeError("Initialization/reference weights changed")
        diagnostic={"kind":"DIAGNOSTIC_NOT_INSTANCE_SUPERVISION","steps":records,
                    "tracked_max_parameter_changes":changes,"reference_unchanged":reference_unchanged,
                    "seconds":time.perf_counter()-started,"no_accuracy_claim":True}
        write_json(out/"optimizer_diagnostic.json",diagnostic)
        with (out/f"diagnostic_after_{args.diagnostic_steps}steps_state_dict.pt").open("xb") as stream:
            torch.save(small.state_dict(),stream)
        print("OPTIMIZER_DIAGNOSTIC",json.dumps(diagnostic),flush=True)
    write_json(out/"completion.json",{"status":"PASS","REAL_INTEGRATION":"NOT RUN",
                                      "total_seconds":time.perf_counter()-start,
                                      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    return 0


if __name__=="__main__":
    raise SystemExit(main())
