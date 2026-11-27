import java.io.*;
import java.util.*;
public final class Runner {
    private static StringTokenizer t;
    private static int integer() {
        return Integer.parseInt(t.nextToken());
    }
    private static long value() {
        return Long.parseLong(t.nextToken());
    }
    private static long[] values(int n) {
        long[] a = new long[n];
        for (int i = 0; i < n; i++)
            a[i] = value();
        return a;
    }
    private static String json(Object v) {
        if (v == null)
            return "null";
        if (v instanceof long[] a)
            return Arrays.toString(a);
        if (v instanceof int[] a)
            return Arrays.toString(a);
        if (v instanceof double[] a)
            return Arrays.toString(a);
        if (v instanceof List<?> a) {
            StringJoiner j = new StringJoiner(",", "[", "]");
            for (Object x : a)
                j.add(json(x));
            return j.toString();
        }
        return v.toString();
    }
    private static Object execute(String line) {
        t = new StringTokenizer(line);
        String op = t.nextToken();
        if (Set.of("sort", "next", "lower", "top", "top_heap", "window").contains(op)) {
            int n = integer();
            long k = Set.of("lower", "top", "top_heap", "window").contains(op) ? value() : 0;
            long[] a = values(n);
            return switch (op) {
                case "sort" -> Algorithms.mergeSort(a);
                case "next" -> Algorithms.nextGreaterIndex(a);
                case "lower" -> Algorithms.lowerBound(a, k);
                case "top" -> Algorithms.topK(a, (int)k);
                case "top_heap" -> Algorithms.topKHeap(a, (int)k);
                default -> Algorithms.windowMax(a, (int)k);
            };
        }
        if (op.equals("list")) {
            int n = integer(), cycle = integer();
            Algorithms.ListNode[] a = new Algorithms.ListNode[n];
            for (int i = 0; i < n; i++)
                a[i] = new Algorithms.ListNode(value());
            for (int i = 0; i + 1 < n; i++)
                a[i].next = a[i + 1];
            var mid = Algorithms.middleNode(n > 0 ? a[0] : null);
            int m = -1, c = -1;
            for (int i = 0; i < n; i++)
                if (a[i] == mid)
                    m = i;
            if (cycle >= 0)
                a[n - 1].next = a[cycle];
            var entry = Algorithms.cycleEntry(n > 0 ? a[0] : null);
            for (int i = 0; i < n; i++)
                if (a[i] == entry)
                    c = i;
            return new int[] {m, c};
        }
        if (op.equals("tree")) {
            int n = integer();
            Algorithms.TreeNode[] a = new Algorithms.TreeNode[n];
            int[][] links = new int[n][2];
            for (int i = 0; i < n; i++) {
                a[i] = new Algorithms.TreeNode(value());
                links[i][0] = integer();
                links[i][1] = integer();
            }
            for (int i = 0; i < n; i++) {
                a[i].left = links[i][0] < 0 ? null : a[links[i][0]];
                a[i].right = links[i][1] < 0 ? null : a[links[i][1]];
            }
            var root = n > 0 ? a[0] : null;
            List<Object> out = new ArrayList<>(Algorithms.treeOrders(root));
            out.add(Algorithms.levelOrder(root));
            return out;
        }
        if (op.equals("lru")) {
            var c = new Algorithms.LRUCache(integer());
            int n = integer();
            List<Long> out = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int kind = integer();
                long key = value(), v = value();
                if (kind == 0)
                    c.put(key, v);
                else
                    out.add(c.get(key));
            }
            return out;
        }
        if (op.equals("bip") || op.equals("topo")) {
            int n = integer(), m = integer();
            int[][] edges = new int[m][2];
            for (int[] e : edges) {
                e[0] = integer();
                e[1] = integer();
            }
            return op.equals("bip") ? Algorithms.isBipartite(n, edges)
                                    : Algorithms.topologicalSort(n, edges);
        }
        if (op.equals("dijkstra")) {
            int n = integer(), m = integer(), s = integer();
            var edges = new Algorithms.WeightedEdge[m];
            for (int i = 0; i < m; i++)
                edges[i] = new Algorithms.WeightedEdge(integer(), integer(), value());
            return Algorithms.dijkstra(n, edges, s);
        }
        if (op.equals("kmp") || op.equals("edit")) {
            int n = integer(), m = integer();
            long[] a = values(n), b = values(m);
            return op.equals("kmp") ? Algorithms.kmpSearch(a, b) : Algorithms.editDistance(a, b);
        }
        if (op.equals("trie")) {
            int n = integer();
            var trie = new Algorithms.Trie();
            List<Boolean> out = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int kind = integer(), size = integer();
                long[] word = values(size);
                if (kind == 0)
                    trie.insert(word);
                else
                    out.add(kind == 1 ? trie.contains(word) : trie.startsWith(word));
            }
            return out;
        }
        if (op.equals("polygon")) {
            int n = integer();
            var q = new Algorithms.Point(value(), value());
            var p = new Algorithms.Point[n];
            for (int i = 0; i < n; i++)
                p[i] = new Algorithms.Point(value(), value());
            return new long[] {Algorithms.polygonArea2(p), Algorithms.pointInPolygon(q, p)};
        }
        if (op.equals("comb"))
            return Algorithms.combinations(integer(), integer());
        if (op.equals("comb_backtrack"))
            return Algorithms.combinationsBacktrack(integer(), integer());
        if (op.equals("sqrt"))
            return Algorithms.sqrtNewton(Double.parseDouble(t.nextToken()));
        if (op.equals("gauss")) {
            int n = integer();
            double[][] a = new double[n][n];
            double[] b = new double[n];
            for (double[] row : a)
                for (int c = 0; c < n; c++)
                    row[c] = Double.parseDouble(t.nextToken());
            for (int i = 0; i < n; i++)
                b[i] = Double.parseDouble(t.nextToken());
            return Algorithms.gaussianSolve(a, b);
        }
        if (op.equals("dsu")) {
            int n = integer(), q = integer();
            var d = new Algorithms.DSU(n);
            List<Boolean> out = new ArrayList<>();
            for (int i = 0; i < q; i++) {
                int kind = integer(), a = integer(), b = integer();
                out.add(kind == 0 ? d.union(a, b) : d.find(a) == d.find(b));
            }
            return out;
        }
        if (op.equals("fenwick")) {
            int n = integer(), q = integer();
            var f = new Algorithms.Fenwick(n);
            List<Long> out = new ArrayList<>();
            for (int i = 0; i < q; i++) {
                int kind = integer(), index = integer();
                long v = value();
                if (kind == 0)
                    f.add(index, v);
                else
                    out.add(f.prefixSum(index));
            }
            return out;
        }
        throw new IllegalArgumentException("unknown operation");
    }
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = br.readLine()) != null) {
            try {
                System.out.println(json(execute(line)));
            } catch (IllegalArgumentException | ArithmeticException | IndexOutOfBoundsException e) {
                System.out.println("{\"error\":true}");
            }
        }
    }
}
