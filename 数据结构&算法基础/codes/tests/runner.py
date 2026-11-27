"""Line protocol adapter; not used inside timing measurements."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import algorithms as a

def execute(line):
    tokens=iter(line.split());op=next(tokens)
    def integer():return int(next(tokens))
    def values(n):return [integer() for _ in range(n)]
    if op in ('sort','next','lower','top','top_heap','window'):
        n=integer();k=integer() if op in ('lower','top','top_heap','window') else 0;v=values(n)
        return {'sort':lambda:a.merge_sort(v),'next':lambda:a.next_greater_index(v),'lower':lambda:a.lower_bound(v,k),'top':lambda:a.top_k(v,k),'top_heap':lambda:a.top_k_heap(v,k),'window':lambda:a.window_max(v,k)}[op]()
    if op=='list':
        n,cycle=integer(),integer();nodes=[a.ListNode(v) for v in values(n)]
        for i in range(n-1):nodes[i].next=nodes[i+1]
        middle=a.middle_node(nodes[0] if n else None)
        if cycle>=0:nodes[-1].next=nodes[cycle]
        entry=a.cycle_entry(nodes[0] if n else None)
        return [nodes.index(middle) if middle else -1,nodes.index(entry) if entry else -1]
    if op=='tree':
        n=integer();rows=[values(3) for _ in range(n)];nodes=[a.TreeNode(r[0]) for r in rows]
        for i,(_,l,r) in enumerate(rows):nodes[i].left=nodes[l] if l>=0 else None;nodes[i].right=nodes[r] if r>=0 else None
        root=nodes[0] if n else None
        return [*a.tree_orders(root),a.level_order(root)]
    if op=='lru':
        cache=a.LRUCache(integer());out=[]
        for _ in range(integer()):
            kind,key,value=integer(),integer(),integer()
            if kind==0:cache.put(key,value)
            else:out.append(cache.get(key))
        return out
    if op in ('bip','topo'):
        n,m=integer(),integer();e=[tuple(values(2)) for _ in range(m)]
        return a.is_bipartite(n,e) if op=='bip' else a.topological_sort(n,e)
    if op=='dijkstra':
        n,m,s=integer(),integer(),integer();return a.dijkstra(n,[tuple(values(3)) for _ in range(m)],s)
    if op in ('kmp','edit'):
        n,m=integer(),integer();x,y=values(n),values(m)
        return a.kmp_search(x,y) if op=='kmp' else a.edit_distance(x,y)
    if op=='trie':
        tree=a.Trie();out=[]
        for _ in range(integer()):
            kind,n=integer(),integer();word=values(n)
            if kind==0:tree.insert(word)
            else:out.append(tree.contains(word) if kind==1 else tree.starts_with(word))
        return out
    if op=='polygon':
        n=integer();point=tuple(values(2));poly=[tuple(values(2)) for _ in range(n)]
        return [a.polygon_area2(poly),a.point_in_polygon(point,poly)]
    if op=='comb':return a.combinations(integer(),integer())
    if op=='comb_backtrack':return a.combinations_backtrack(integer(),integer())
    if op=='sqrt':return a.sqrt_newton(float(next(tokens)))
    if op=='gauss':
        n=integer();mat=[[float(next(tokens)) for _ in range(n)] for _ in range(n)];b=[float(next(tokens)) for _ in range(n)]
        return a.gaussian_solve(mat,b)
    if op=='dsu':
        d=a.DSU(integer());out=[]
        for _ in range(integer()):
            kind,x,y=integer(),integer(),integer();out.append(d.union(x,y) if kind==0 else d.find(x)==d.find(y))
        return out
    if op=='fenwick':
        f=a.Fenwick(integer());out=[]
        for _ in range(integer()):
            kind,x,y=integer(),integer(),integer()
            if kind==0:f.add(x,y)
            else:out.append(f.prefix_sum(x))
        return out
    raise ValueError('unknown operation')

if __name__=='__main__':
    for line in sys.stdin:
        try:print(json.dumps(execute(line),allow_nan=False,separators=(',',':')))
        except (ValueError,ArithmeticError,IndexError):print('{"error":true}')
