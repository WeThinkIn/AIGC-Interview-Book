"""Single-process microbenchmarks: input creation and checks are outside timed regions."""
import sys,time,statistics,json,heapq
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import algorithms as a
N=32768
def input_values():
    state=20261127;out=[]
    for _ in range(N):
        state=(1664525*state+1013904223)&0xffffffff;out.append(state%100000)
    return out
def measure(task,method,fn,expected):
    assert fn()==expected
    for _ in range(8):fn()
    samples=[];sink=0
    for _ in range(11):
        t=time.perf_counter_ns();r=fn();samples.append((time.perf_counter_ns()-t)/1e6)
        sink^=r[0] if r else 0
    print(json.dumps({'task':task,'method':method,'samples_ms':samples,'median_ms':statistics.median(samples),'sink':sink}),flush=True)
def main():
    a0=input_values()
    for name,values in [('random',a0),('sorted',sorted(a0)),('reversed',sorted(a0,reverse=True)),('duplicates',[x%8 for x in a0])]:
        expected=sorted(values)
        measure('sort_'+name,'merge_buffers',lambda:a.merge_sort(values),expected)
        measure('sort_'+name,'builtin_sorted',lambda:sorted(values),expected)
    for k in (8,128,N//2):
        expected=sorted(a0,reverse=True)[:k]
        measure('top_'+str(k),'bounded_heap',lambda:a.top_k_heap(a0,k),expected)
        measure('top_'+str(k),'adaptive',lambda:a.top_k(a0,k),expected)
        measure('top_'+str(k),'heapq_nlargest',lambda:heapq.nlargest(k,a0),expected)
        measure('top_'+str(k),'full_sort',lambda:sorted(a0,reverse=True)[:k],expected)
    k=128;expected=[max(a0[i:i+k]) for i in range(N-k+1)]
    measure('window_128','monotonic_deque',lambda:a.window_max(a0,k),expected)
    measure('window_128','scan_each_window',lambda:[max(a0[i:i+k]) for i in range(N-k+1)],expected)
if __name__=='__main__':main()
