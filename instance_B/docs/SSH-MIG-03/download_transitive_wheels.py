import concurrent.futures,json,pathlib,re,subprocess,sys
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
b=pathlib.Path(__file__).resolve().parents[2]; d=b/'docs/SSH-MIG-03'
tags={t:i for i,t in enumerate(sys_tags())}
names={'filelock','fsspec','jinja2','markupsafe','mpmath','networkx','sympy'}
pins=[m.groups() for m in re.finditer(r'^([A-Za-z0-9_-]+)==([^\s\\]+)',(d/'requirements.lock').read_text(),re.M) if m[1].lower() in names]
assert len(pins)==7,pins
def work(pair):
 name,ver=pair; u='https://pypi.org/pypi/'+name+'/'+ver+'/json'
 m=json.loads(subprocess.check_output(['curl','-4','-fsSL','--connect-timeout','10','--max-time','90',u]))
 candidates=[]
 for f in m['urls']:
  if not f['filename'].endswith('.whl'): continue
  _,_,_,wt=parse_wheel_filename(f['filename']); ranks=[tags[t] for t in wt if t in tags]
  if ranks: candidates.append((min(ranks),f))
 f=min(candidates,key=lambda x:x[0])[1]
 record={k:f[k] for k in ['filename','url','digests','size']}; record['metadata_url']=u
 (d/(name+'_wheel_source.json')).write_text(json.dumps(record,indent=2)+'\n')
 subprocess.run(['/usr/bin/python3','-I','-B','-S',str(d/'download_official.py'),f['url'],str(b/'.cache/wheelhouse'/f['filename']),str(f['size']),f['digests']['sha256'],'524288'],check=True)
 print(name,ver,'VERIFIED',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool: list(pool.map(work,pins))
