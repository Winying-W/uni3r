import concurrent.futures, hashlib, json, math, pathlib, subprocess, sys, time
url,out,size,digest=sys.argv[1:5]; out=pathlib.Path(out); size=int(size)
root=pathlib.Path(__file__).resolve().parents[2]
assert out.resolve().is_relative_to(root)
assert url.startswith(('https://files.pythonhosted.org/','https://download.pytorch.org/','https://download-r2.pytorch.org/','https://github.com/astral-sh/'))
parts=out.parent/(out.name+'.parts'); parts.mkdir(exist_ok=True)
chunk=int(sys.argv[5]) if len(sys.argv)>5 else 8*1024*1024
started=time.time()
def fetch(i):
 start=i*chunk; end=min(size-1,start+chunk-1); p=parts/str(i)
 if p.exists() and p.stat().st_size==end-start+1: return
 for attempt in range(3):
  res=subprocess.run(['curl','-4','-fsSL','--connect-timeout','15','--max-time','180','--range',str(start)+'-'+str(end),url,'-o',str(p)],capture_output=True,text=True)
  if res.returncode==0 and p.stat().st_size==end-start+1: return
 raise RuntimeError('Chunk failed '+str(i)+' '+res.stderr[:200])
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
 for count,_ in enumerate(pool.map(fetch,range(math.ceil(size/chunk))),1):
  if count%16==0: print('CHUNKS',count,'/',math.ceil(size/chunk),flush=True)
with out.open('xb') as f:
 for i in range(math.ceil(size/chunk)): f.write((parts/str(i)).read_bytes())
hashval=hashlib.file_digest(out.open('rb'),'sha256').hexdigest()
assert hashval==digest,(hashval,digest)
print('DOWNLOAD_VERIFIED',size,'seconds',round(time.time()-started,1),flush=True)
