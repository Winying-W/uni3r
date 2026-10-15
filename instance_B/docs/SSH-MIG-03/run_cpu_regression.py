import json, os, pathlib, subprocess, sys, datetime
r=pathlib.Path(__file__).resolve().parents[3]; b=r/"instance_B"; d=pathlib.Path(__file__).resolve().parent
assert sys.executable == str(b/".venv/bin/python")
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
def run(cmd,name):
    with (d/name).open("x") as f:
        p=subprocess.run(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT)
    print(name,"exit",p.returncode,flush=True)
    if p.returncode: raise SystemExit(p.returncode)
run([sys.executable,"-B",str(d/"verify_environment.py")],"environment_verification.json")
run([str(b/".tools/uv"),"pip","check","--python",sys.executable],"dependency_check.txt")
files=["test_contracts.py","test_student.py","test_matcher.py","test_criterion.py","test_b02_integration.py"]
run([sys.executable,"-B","-m","pytest","--collect-only","-q","-p","no:cacheprovider",*[str(b/"tests"/n) for n in files]],"pytest_collection.txt")
base=b/"results/B-02/SSH-MIG-03"
assert not base.exists()
for mode in ["unit","integration"]:
    run([sys.executable,"-B","-m","instance_B.run_b02_checks","--mode",mode,"--run-dir",str(base/mode)],mode+"_console.txt")
records=[]; comparison={}
for mode,old in [("unit","unit_01"),("integration","integration_01")]:
    result=json.loads((base/mode/"unit_results.json").read_text())
    hist=json.loads((b/"results/B-02/20261008T130554Z_cpu_b02"/old/"unit_results.json").read_text())
    ids=sorted(x["test"] for x in result["records"]); oldids=sorted(x["test"] for x in hist["records"])
    comparison[mode]={"same_test_ids":ids==oldids,"actual_count":result["tests_run"],"historical_count":hist["tests_run"]}
    records+=result["records"]
summary={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"collected":len(records),"pass":sum(x["status"]=="PASS" for x in records),"skip":sum(x["status"]=="SKIP" for x in records),"fail":sum(x["status"] not in ("PASS","SKIP") for x in records),"test_set_comparison":comparison,"records":records,"GPU_USED":"NO","optimizer_steps":0,"new_training_checkpoints":0,"evidence_kind":"CPU_SYNTHETIC_REGRESSION_NOT_REAL_AC_INTEGRATION"}
(d/"regression_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="records"},indent=2),flush=True)
assert summary["collected"]==76 and summary["pass"]==75 and summary["skip"]==1 and summary["fail"]==0
assert all(x["same_test_ids"] for x in comparison.values())
