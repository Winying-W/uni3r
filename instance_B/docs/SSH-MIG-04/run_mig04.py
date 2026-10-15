import gc,json,os,pathlib,sys,time,traceback,importlib.metadata as metadata
r=pathlib.Path(__file__).resolve().parents[3]; b=r/'instance_B'; d=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(r))
assert sys.executable==str(b/'.venv/bin/python')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
import torch,scipy,numpy
from instance_B import run_b03_checks as old
from instance_B.b03_utils import tensor_fingerprints,tree_digest,file_sha256
from instance_B.training import BudgetLedger,TrainingEngine
from instance_B.student import InstanceStudent,load_dpt_initialization
from instance_B.checkpointing import save_checkpoint,load_verified_payload,validate_payload,restore_checkpoint,environment
from instance_B.probe_m0 import inspect_c_handoff
from uni3r.instance.contracts import validate_frozen_scene_batch
out=b/'results/B-03/SSH-MIG-04/attempt_01'
def save(name,obj):
 with (d/name).open('x') as f: json.dump(obj,f,indent=2,allow_nan=False); f.write('\n')
def progress(x): print(x,flush=True)
assert torch.__version__=='2.1.0+cu121' and scipy.__version__=='1.15.3' and numpy.__version__=='1.26.4'
torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
assert not torch.cuda.is_initialized()
versions={v.metadata['Name']:v.version for v in metadata.distributions()}
prior=json.loads((b/'docs/SSH-MIG-03/environment_versions.json').read_text())
assert versions==prior['packages'], 'Environment changed since MIG03'
save('environment.json',{'environment':environment(),'packages':versions,'matches_MIG03':True,'CUDA_VISIBLE_DEVICES':'','GPU_USED':'NO','tmp':os.environ['TMPDIR'],'cache':os.environ['XDG_CACHE_HOME']})
ledger=BudgetLedger(b/'results/B-03/20261008T140224Z_cpu_b03/optimizer_budget.jsonl')
assert ledger.physical_calls==5
# Read-only A cache admission; no model constructor/forward here.
ap=r/'instance_A/results/M0_20261008_01'
receipt=json.loads((ap/'report.json').read_text()); ameta=json.loads((ap/'metadata.json').read_text())
ah={n:file_sha256(ap/n) for n in ['fixture.pt','metadata.json','source_manifest.json']}
assert all(v==receipt['artifact_sha256'][n] for n,v in ah.items())
assert receipt['status']=='A_M0_REAL_FIXTURE_VERIFIED' and receipt['purpose']=='M0_DIAGNOSTIC_ONLY_NOT_FORMAL_TRAIN_OR_VAL'
batch=torch.load(ap/'fixture.pt',map_location='cpu',weights_only=True)
assert batch['metadata']==ameta; validate_frozen_scene_batch(batch)
assert ameta['evidence_kind']=='real'
save('A_cache_current.json',{'status':'PASS_READABLE_HASH_AND_SHARED_SCHEMA','path':str(ap/'fixture.pt'),'bytes':(ap/'fixture.pt').stat().st_size,'hashes':ah,'scene_id':ameta['scene_id'],'frame_ids':ameta['frame_ids'],'purpose':receipt['purpose'],'images_shape':list(batch['images'].shape),'token_shapes':{str(k):list(v.shape) for k,v in batch['patch_tokens'].items()},'deserialization':'weights_only=True map_location=cpu','new_model_forward':False,'online_A_export_rerun':False,'external_source_paths_followed':False,'historical_forward':'PASS in B-03 attempt_01/m0_probe/real_student_forward.json; not rerun this round'})
del batch; gc.collect()
c=inspect_c_handoff(); save('C_current.json',c)
progress('A cache current read/hash/schema PASS; C='+c['status'])
initdir=b/'results/B-01/20261008_cpu_01'
initialization=load_dpt_initialization(initdir/'dpt_init_from_2views.pt',initdir/'dpt_init_receipt.json')
initial_fingerprints=tensor_fingerprints(dict(initialization.state))
old.test_student.load_init=lambda:initialization
old.test_training.CONTEXT.update(initialization=initialization,ledger=ledger,out=out)
assert old.suite([old.test_contracts,old.test_student,old.test_matcher,old.test_criterion,old.test_b02_integration],out,'regression')
assert old.suite([old.test_training],out,'training_tests')
assert ledger.physical_calls==5
progress('Original B02 and 8 B03 training tests PASS; optimizer calls remain 5')
# Original schedule prefix W1, I, W2: three new calls total, including resumed W2.
# No original model/engine/interface/config/assertion is modified.
torch.manual_seed(7303)
generators={'data':torch.Generator().manual_seed(73031),'eval':torch.Generator().manual_seed(73039)}
samples=json.loads((b/'results/B-03/20261008T140224Z_cpu_b03/attempt_01/synthetic_sample_manifest.json').read_text())
evaluation=old.test_training.make_example('complete',generators['eval'],'FIXED_EVAL')[0]
model=InstanceStudent(old.test_student.small_config(),dpt_init=initialization)
engine=TrainingEngine(model,ledger,event_log=out/'continuous_events.jsonl')
legacy=b/'results/B-03/20261008T140224Z_cpu_b03/attempt_01/synthetic_engine_W1.pt'
payload,legacy_receipt=load_verified_payload(legacy,legacy.with_suffix('.pt.receipt.json'))
try:
 validate_payload(payload,engine,samples,generators)
except ValueError as exc:
 assert str(exc)=='Exact resume environment mismatch',str(exc)
 save('legacy_checkpoint_compatibility.json',{'status':'EXPECTED_STRICT_REJECTION','error':str(exc),'old_environment':payload['environment'],'current_environment':environment(),'old_checkpoint_sha256':legacy_receipt['sha256'],'guard_relaxed':False,'old_payload_restored':False})
else: raise AssertionError('Expected exact Python environment mismatch')
del payload; gc.collect()
first=old.next_window(engine,generators,0); first_hash=tree_digest(first)
w1=engine.run_window(first,label='SSH_MIG04_CONTINUOUS_W1'); assert tree_digest(first)==first_hash
assert ledger.physical_calls==6
save('W1_update.json',w1)
saved_outputs=old.evaluate(model,evaluation)
path=out/'synthetic_migration_W1.pt'; receipt=save_checkpoint(engine,path,samples,generators)
saved=old.engine_snapshot(engine,generators)
payload,_=load_verified_payload(path,path.with_suffix('.pt.receipt.json'))
old.test_checkpointing.CONTEXT.update(payload=payload,engine=engine,samples=samples,generators=generators,out=out,path=path,receipt=receipt,initialization=initialization)
assert old.suite([old.test_checkpointing],out,'checkpoint_tests')
old.test_checkpointing.CONTEXT.clear(); del payload; gc.collect()
witness=torch.rand(4)
ignored=old.next_window(engine,generators,1)
before=old.engine_snapshot(engine,generators); calls=ledger.physical_calls
skipped=engine.run_window(ignored,label='SSH_MIG04_CONTINUOUS_I'); after=old.engine_snapshot(engine,generators)
for key in ['student','optimizer_tensors','optimizer_digest','groups','torch_rng','generator_rng']: assert before[key]==after[key],key
assert before['state']['global_update_step']==after['state']['global_update_step']==1
assert calls==ledger.physical_calls and skipped['forward_count']==0 and engine.optimizer.state
save('skip_after_moments.json',{'status':'PASS','before_counters':before['state'],'after_counters':after['state'],'parameters_buffers_optimizer_LR_RNG_unchanged':True,'optimizer_calls_unchanged':True,'report':skipped})
second=old.next_window(engine,generators,2); second_hash=tree_digest(second)
w2=engine.run_window(second,label='SSH_MIG04_CONTINUOUS_W2'); assert ledger.physical_calls==7
continuous=old.engine_snapshot(engine,generators); final_outputs=old.evaluate(model,evaluation)
candidate,_=restore_checkpoint(path,path.with_suffix('.pt.receipt.json'),engine,initialization,samples,generators,event_log=out/'resumed_events.jsonl')
assert old.engine_snapshot(candidate,generators)==saved
restored=old.evaluate(candidate.model,evaluation)
for key in saved_outputs: torch.testing.assert_close(saved_outputs[key],restored[key],atol=0,rtol=0)
assert torch.equal(torch.rand(4),witness)
resumed_ignored=old.next_window(candidate,generators,1); assert tree_digest(resumed_ignored)==tree_digest(ignored)
candidate.run_window(resumed_ignored,label='SSH_MIG04_RESUMED_I')
resumed_second=old.next_window(candidate,generators,2); assert tree_digest(resumed_second)==second_hash
rw2=candidate.run_window(resumed_second,label='SSH_MIG04_RESUMED_W2'); assert rw2['losses']==w2['losses']
resumed=old.engine_snapshot(candidate,generators); assert resumed==continuous
outputs=old.evaluate(candidate.model,evaluation); errors={}
for key in final_outputs:
 torch.testing.assert_close(outputs[key],final_outputs[key],atol=0,rtol=0)
 errors[key]=(outputs[key]-final_outputs[key]).abs().max().item()
assert ledger.physical_calls==8
assert tensor_fingerprints(dict(initialization.state))==initial_fingerprints
assert not torch.cuda.is_initialized()
save('checkpoint_recovery.json',{'status':'PASS_NEW_ENV_SAVE_RESTORE_CONTINUATION','new_checkpoint':str(path),'checkpoint_bytes':receipt['bytes'],'checkpoint_sha256':receipt['sha256'],'purpose':receipt['purpose'],'executed_original_schedule_prefix':['W1','I','W2'],'W3':'NOT_RERUN to retain immutable global 8-call budget','original_five_call_diagnostic_rerun':False,'new_optimizer_calls':3,'global_ledger_calls_before':5,'global_ledger_calls_after':8,'continuous_logical_updates':engine.state['global_update_step'],'resumed_logical_updates':candidate.state['global_update_step'],'saved_state_outputs_exact':True,'final_student_optimizer_counters_cursor_LR_RNG_exact':True,'final_raw_output_max_errors':errors,'initialization_unchanged':True,'GPU_USED':'NO','real_training':'NOT_RUN'})
records=[]; suites={}
for label,historical in [('regression','regression_results.json'),('training_tests','training_tests_results.json'),('checkpoint_tests','checkpoint_tests_results.json')]:
 v=json.loads((out/(label+'_results.json')).read_text()); oldv=json.loads((b/'results/B-03/20261008T140224Z_cpu_b03/attempt_01'/historical).read_text())
 assert sorted(x['test'] for x in v['records'])==sorted(x['test'] for x in oldv['records'])
 records+=v['records']; suites[label]={k:v[k] for k in ['tests_run','failures','errors','skips','seconds']}; suites[label]['same_historical_test_ids']=True
save('test_summary.json',{'suites':suites,'collected':len(records),'PASS':sum(x['status']=='PASS' for x in records),'SKIP':sum(x['status']=='SKIP' for x in records),'FAIL':sum(x['status'] not in ('PASS','SKIP') for x in records),'records':records,'new_synthetic_CPU_optimizer_calls':3,'real_data_optimizer_calls':0,'GPU_USED':'NO'})
progress('MIG04_CPU_REGRESSION_PASS: 88 collected, 87 PASS, 1 SKIP; 3 new synthetic CPU steps; ledger now 8/8; C='+c['status'])
