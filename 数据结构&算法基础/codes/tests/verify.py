"""Deterministic boundary, randomized differential, and three-language checks.

python tests/verify.py --java /path/bin/java --javac /path/bin/javac --out /tmp/algorithms-check
No third-party Python packages required. Compilation products stay outside codes/.
"""
import argparse,bisect,collections,itertools,json,math,os,platform,random,subprocess,sys,time
from pathlib import Path

CODES=Path(__file__).resolve().parents[1]
SEED=20261127
rng=random.Random(SEED)
CASES=[]
def flat(x):
    if isinstance(x,(tuple,list)):
        return [s for y in x for s in flat(y)]
    return [str(x)]
def case(op,args,expected):CASES.append({'op':op,'line':' '.join([op,*flat(args)]),'expected':expected})
def edit_ref(a,b):
    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):d[i][0]=i
    for j in range(len(b)+1):d[0][j]=j
    for i,x in enumerate(a,1):
        for j,y in enumerate(b,1):d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(x!=y))
    return d[-1][-1]
def tree_ref(rows):
    out=[[],[],[],[]]
    def dfs(i,depth):
        if i<0:return
        if len(out[3])<=depth:out[3].append([])
        v,l,r=rows[i];out[3][depth].append(v);out[0].append(v);dfs(l,depth+1);out[1].append(v);dfs(r,depth+1);out[2].append(v)
    if rows:dfs(0,0)
    return out
def generate():
    for _ in range(220):
        n=rng.randrange(0,160);a=[rng.randrange(-100,101) for _ in range(n)]
        if _<6:a=[[],[1],[1,1],[2,1],[-(1<<63),(1<<63)-1],[3,2,1]][_];n=len(a)
        case('sort',[n,a],sorted(a))
        case('next',[n,a],[next((j for j in range(i+1,n) if a[j]>a[i]),-1) for i in range(n)])
        k=rng.randrange(n+8);case('top',[n,k,a],sorted(a,reverse=True)[:k])
        case('top_heap',[n,k,a],sorted(a,reverse=True)[:k])
        b=sorted(a);target=rng.randrange(-110,111);case('lower',[n,target,b],bisect.bisect_left(b,target))
        if n:
            k=rng.randrange(1,n+1);case('window',[n,k,a],[max(a[i:i+k]) for i in range(n-k+1)])
        cycle=rng.randrange(-1,n) if n else -1
        case('list',[n,cycle,a],[n//2 if n else -1,cycle])
    for _ in range(100):
        n=rng.randrange(80);rows=[[rng.randrange(-100,100),-1,-1] for _ in range(n)]
        slots=[(0,1),(0,2)] if n else []
        for i in range(1,n):
            p,side=slots.pop(rng.randrange(len(slots)));rows[p][side]=i;slots.extend([(i,1),(i,2)])
        case('tree',[n,rows],tree_ref(rows))
        capacity=rng.randrange(8);cache={};order=[];out=[];events=[]
        for i in range(100):
            kind=rng.randrange(2);key=rng.randrange(12);value=rng.randrange(-3,4);events.append([kind,key,value])
            if kind:
                out.append(cache.get(key))
                if key in cache:order.remove(key);order.append(key)
            elif capacity:
                if key in cache:order.remove(key)
                cache[key]=value;order.append(key)
                if len(order)>capacity:del cache[order.pop(0)]
        case('lru',[capacity,len(events),events],out)
    for _ in range(160):
        n=rng.randrange(0,9);edges=[(rng.randrange(n),rng.randrange(n)) for _ in range(rng.randrange(20))] if n else []
        bip=any(all(((mask>>u)&1)!=((mask>>v)&1) for u,v in edges) for mask in range(1<<n))
        case('bip',[n,len(edges),edges],bip)
        # A tiny Kahn reference also asserts edge precedence independently after output.
        degree=[sum(v==i for u,v in edges) for i in range(n)];q=collections.deque(i for i in range(n) if degree[i]==0);order=[]
        while q:
            u=q.popleft();order.append(u)
            for x,v in edges:
                if x==u:
                    degree[v]-=1
                    if degree[v]==0:q.append(v)
        case('topo',[n,len(edges),edges],order if len(order)==n else None)
        if n:
            e=[(u,v,rng.randrange(30)) for u,v in edges];s=rng.randrange(n);dist=[(1<<63)-1]*n;dist[s]=0
            for _pass in range(n-1):
                for u,v,w in e:
                    if dist[u]!=(1<<63)-1:dist[v]=min(dist[v],dist[u]+w)
            case('dijkstra',[n,len(e),s,e],dist)
    for _ in range(200):
        a=[rng.choice([-3,0,1,2,0x4e2d,0x1f600]) for _ in range(rng.randrange(45))]
        b=[rng.choice([-3,0,1,2,0x4e2d,0x1f600]) for _ in range(rng.randrange(12))]
        hit=next((i for i in range(len(a)-len(b)+1) if a[i:i+len(b)]==b),-1)
        case('kmp',[len(a),len(b),a,b],hit)
        case('edit',[len(a),len(b),a,b],edit_ref(a,b))
    for _ in range(40):
        words=set();events=[];out=[]
        for i in range(100):
            kind=rng.randrange(3);word=tuple(rng.randrange(-2,4) for _ in range(rng.randrange(7)));events.append([kind,len(word),word])
            if kind==0:words.add(word)
            elif kind==1:out.append(word in words)
            else:out.append(not word or any(w[:len(word)]==word for w in words))
        case('trie',[len(events),events],out)
    for x in range(-3,7):
        for y in range(-3,7):
            poly=[(0,0),(4,0),(4,4),(0,4)]
            expected=0 if (x in (0,4) and 0<=y<=4) or (y in (0,4) and 0<=x<=4) else (1 if 0<x<4 and 0<y<4 else -1)
            for p in (poly,poly[::-1]):case('polygon',[len(p),x,y,p],[32,expected])
    # Concave polygon, boundary vertex, horizontal edge, repeated closing vertex.
    for q,e in [((1,1),1),((3,3),-1),((2,2),0),((4,1),0),((1,3),1)]:
        p=[(0,0),(4,0),(4,2),(2,2),(2,4),(0,4),(0,0)];case('polygon',[len(p),q,p],[24,e])
    for n in range(11):
        for k in range(n+3):
            expected=[list(x) for x in itertools.combinations(range(1,n+1),k)]
            case('comb',[n,k],expected);case('comb_backtrack',[n,k],expected)
    case('comb_backtrack',[129,129],{'error':True})
    for x in [0.,-0.,5e-324,1e-320,1e-300,1e-20,1.,2.,1e200,1.7976931348623157e308]+[math.ldexp(rng.random()+.5,rng.randrange(-1020,1023)) for _ in range(150)]:
        case('sqrt',[x],math.sqrt(x))
    case('gauss',[0],[])
    for _ in range(70):
        n=rng.randrange(1,10);a=[[rng.uniform(-1,1) for _ in range(n)] for _ in range(n)]
        for i in range(n):a[i][i]+=n+1
        x=[rng.uniform(-2,2) for _ in range(n)];b=[sum(v*y for v,y in zip(row,x)) for row in a]
        factor=10.**rng.randrange(-150,151);a=[[v*factor for v in row] for row in a];b=[v*factor for v in b]
        if n>1:a[0],a[-1]=a[-1],a[0];b[0],b[-1]=b[-1],b[0]
        case('gauss',[n,a,b],x)
    for _ in range(70):
        n=rng.randrange(1,30);labels=list(range(n));ev=[];out=[];arr=[0]*n;fv=[];fo=[]
        for i in range(100):
            kind=rng.randrange(2);a,b=rng.randrange(n),rng.randrange(n);ev.append([kind,a,b])
            if kind:out.append(labels[a]==labels[b])
            else:
                la,lb=labels[a],labels[b];out.append(la!=lb);labels=[la if x==lb else x for x in labels]
            idx=rng.randrange(n+1) if kind else rng.randrange(n);delta=rng.randrange(-100,101);fv.append([kind,idx,delta])
            if kind:fo.append(sum(arr[:idx]))
            else:arr[idx]+=delta
        case('dsu',[n,len(ev),ev],out);case('fenwick',[n,len(fv),fv],fo)
    invalid=[('window',[0,0]),('window',[2,3,1,2]),('top',[0,-1]),('lru',[-1,0]),('bip',[2,1,0,2]),('topo',[-1,0]),('dijkstra',[2,1,0,0,1,-1]),('dijkstra',[1,1,0,0,0,1<<62]),('comb',[2,-1]),('sqrt',[-1]),('sqrt',['NaN']),('sqrt',['Infinity']),('gauss',[2,1,2,2,4,1,2]),('polygon',[2,0,0,0,0,1,1]),('dsu',[1,1,0,0,1]),('fenwick',[1,1,0,1,5])]
    for op,args in invalid:case(op,args,{'error':True})
    # 20k-deep trees and tries exercise stack safety of implementation and destruction.
    n=20000;rows=[[i,i+1 if i+1<n else -1,-1] for i in range(n)]
    case('tree',[n,rows],[list(range(n)),list(range(n-1,-1,-1)),list(range(n-1,-1,-1)),[[i] for i in range(n)]])
    word=list(range(n));case('trie',[3,0,n,word,1,n,word,2,n-1,word[:-1]],[True,True])

def equal(a,b,op):
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equal(x,y,op) for x,y in zip(a,b))
    if op=='sqrt' and isinstance(a,(float,int)) and isinstance(b,(float,int)):
        return math.isfinite(a) and abs(a-b)<=3*math.ulp(b)
    if op=='gauss' and isinstance(a,(float,int)) and isinstance(b,(float,int)):
        return math.isfinite(a) and math.isclose(a,b,rel_tol=2e-11,abs_tol=2e-11)
    return a==b and (not isinstance(b,bool) or isinstance(a,bool))
def run(cmd,**kwargs):
    r=subprocess.run(cmd,capture_output=True,text=True,**kwargs)
    if r.returncode:raise RuntimeError(str(cmd)+'\n'+r.stdout[-2000:]+r.stderr[-5000:])
    return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--java',default='java');p.add_argument('--javac',default='javac');p.add_argument('--out',required=True);p.add_argument('--cxx',default='clang++');p.add_argument('--skip-sanitizer',action='store_true');args=p.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    generate();payload='\n'.join(x['line'] for x in CASES)+'\n';(out/'cases.json').write_text(json.dumps(CASES))
    cpp=out/'runner_cpp';cppsan=out/'runner_cpp_sanitized'
    flags=['-std=c++20','-Wall','-Wextra','-Wpedantic','-Werror'];compilations=[]
    for dest,extra in [(cpp,['-O3'])]+([] if args.skip_sanitizer else [(cppsan,['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'])]):
        cmd=[args.cxx,*flags,*extra,str(CODES/'tests/runner.cpp'),'-o',str(dest)];r=run(cmd,timeout=100);compilations.append({'cmd':cmd,'stderr':r.stderr})
    java_out=out/'classes';java_out.mkdir(exist_ok=True)
    cmd=[args.javac,'-Xlint:all','-Werror','-d',str(java_out),str(CODES/'java/Algorithms.java'),str(CODES/'tests/Runner.java')];r=run(cmd,timeout=100);compilations.append({'cmd':cmd,'stderr':r.stderr})
    commands={'python':[sys.executable,str(CODES/'tests/runner.py')],'cpp':[str(cpp)],'java':[args.java,'-cp',str(java_out),'Runner']}
    if not args.skip_sanitizer:commands['cpp_sanitized']=[str(cppsan)]
    results={};failures=[]
    for lang,cmd in commands.items():
        start=time.perf_counter();r=run(cmd,input=payload,timeout=180);lines=r.stdout.splitlines();assert len(lines)==len(CASES),(lang,len(lines),len(CASES))
        actual=[]
        for i,(line,c) in enumerate(zip(lines,CASES)):
            try:v=json.loads(line)
            except Exception:v={'invalid_json':line}
            actual.append(v)
            if not equal(v,c['expected'],c['op']):failures.append({'language':lang,'index':i,'case':c,'actual':v})
        (out/(lang+'_actual.json')).write_text(json.dumps(actual));results[lang]={'cases':len(lines),'seconds':time.perf_counter()-start,'stderr':r.stderr}
        print(lang,len(lines),'cases',flush=True)
    report={'seed':SEED,'cases':len(CASES),'operations':dict(collections.Counter(x['op'] for x in CASES)),'languages':results,'compilations':compilations,'platform':platform.platform(),'python':sys.version,'failures':failures,'passed':not failures}
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    if failures:print(json.dumps(failures[:5],indent=2));raise SystemExit(1)
    print('All boundary and differential checks passed.')
if __name__=='__main__':main()
