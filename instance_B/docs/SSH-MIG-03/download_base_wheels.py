import concurrent.futures,json,pathlib,subprocess
b=pathlib.Path(__file__).resolve().parents[2]; d=b/'docs/SSH-MIG-03'
def work(pair):
 name,version=pair
 url='https://pypi.org/pypi/'+name+'/'+version+'/json'
 m=json.loads(subprocess.check_output(['curl','-4','-fsSL','--connect-timeout','10','--max-time','90',url]))
 f=next(f for f in m['urls'] if f['filename'].startswith(name+'-'+version+'-cp310-cp310-manylinux_2_17_x86_64') and f['filename'].endswith('.whl'))
 out=b/'.cache/wheelhouse'/f['filename']
 source={k:f[k] for k in ['filename','url','digests','size']}; source['metadata_url']=url
 (d/(name+'_wheel_source.json')).write_text(json.dumps(source,indent=2)+'\n')
 subprocess.run(['python3','-I','-B','-S',str(d/'download_official.py'),f['url'],str(out),str(f['size']),f['digests']['sha256'],'524288'],check=True)
 print(name,'WHEEL_VERIFIED',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p: list(p.map(work,[('numpy','1.26.4'),('scipy','1.15.3')]))
