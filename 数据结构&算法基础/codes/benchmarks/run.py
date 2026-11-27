import argparse,subprocess,json,sys,platform,statistics,hashlib,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--java',default='java');p.add_argument('--javac',default='javac');args=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
def run(cmd):
 r=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout
cpp=out/'bench_cpp';classes=out/'classes';classes.mkdir(exist_ok=True)
compile_cpp=['clang++','-std=c++20','-O3','-DNDEBUG','-Wall','-Wextra','-Werror',str(root/'benchmarks/bench.cpp'),'-o',str(cpp)]
run(compile_cpp);run([args.javac,'-d',str(classes),str(root/'java/Algorithms.java'),str(root/'benchmarks/Bench.java')])
commands={'python':[sys.executable,str(root/'benchmarks/bench.py')],'cpp':[str(cpp)],'java':[args.java,'-Xms256m','-Xmx512m','-cp',str(classes),'Bench']}
records=[]
for lang,cmd in commands.items():
 for fork in range(3):
  result=run(cmd);(out/(lang+'_'+str(fork)+'.jsonl')).write_text(result)
  records.extend(dict(json.loads(line),language=lang,fork=fork) for line in result.splitlines());print(lang,'fork',fork,'done',flush=True)
grouped={}
for r in records:grouped.setdefault((r['language'],r['task'],r['method']),[]).extend(r['samples_ms'])
summary=[{'language':k[0],'task':k[1],'method':k[2],'median_ms':statistics.median(v),'p10_ms':sorted(v)[int(.1*(len(v)-1))],'p90_ms':sorted(v)[int(.9*(len(v)-1))],'samples':len(v)} for k,v in grouped.items()]
report={'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,'java_version':subprocess.run([args.java,'-version'],capture_output=True,text=True).stderr,'cpu':run(['sysctl','-n','machdep.cpu.brand_string']).strip(),'cpp_version':run(['clang++','--version']),'cpp_command':compile_cpp,'n':32768,'seed':20261127,'rng':'uint32 LCG 1664525*x+1013904223; value modulo 100000','process_forks':3,'samples_per_fork':11,'warmups':{'python':8,'cpp':8,'java':40},'input_copies':'included when required to preserve input','data_generation':'excluded','validation':'full outputs compared before timing each method','environment':'shared workstation, no CPU pinning or isolated load','files':{str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in root.rglob('*') if x.is_file() and x.suffix in ('.py','.hpp','.cpp','.java')},'results':summary}
(out/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('benchmark complete',len(summary),'rows')
