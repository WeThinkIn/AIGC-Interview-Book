import java.util.*;

/** Original Java 21 implementations. See ../README.md for domains and ownership. */
public final class Algorithms {
    private Algorithms() {}
    public static final long INF = Long.MAX_VALUE;
    public static final long DIST_LIMIT = (1L << 62) - 1;
    private static void require(boolean ok, String why) {
        if (!ok)
            throw new IllegalArgumentException(why);
    }
    public static final class ListNode {
        public final long value;
        public ListNode next;
        public ListNode(long value) {
            this.value = value;
        }
    }
    public static ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
    public static ListNode cycleEntry(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                slow = head;
                while (slow != fast) {
                    slow = slow.next;
                    fast = fast.next;
                }
                return slow;
            }
        }
        return null;
    }
    public static final class LRUCache {
        private final int capacity;
        private final LinkedHashMap<Long, Long> items = new LinkedHashMap<>(16, .75f, true);
        public LRUCache(int capacity) {
            require(capacity >= 0, "negative capacity");
            this.capacity = capacity;
        }
        public Long get(long key) {
            return items.get(key);
        }
        public void put(long key, long value) {
            if (capacity == 0)
                return;
            items.put(key, value);
            if (items.size() > capacity) {
                Iterator<Long> it = items.keySet().iterator();
                it.next();
                it.remove();
            }
        }
    }
    public static long[] nextGreaterIndex(long[] a) {
        long[] out = new long[a.length];
        Arrays.fill(out, -1);
        int[] stack = new int[a.length];
        int size = 0;
        for (int i = 0; i < a.length; i++) {
            while (size > 0 && a[stack[size - 1]] < a[i])
                out[stack[--size]] = i;
            stack[size++] = i;
        }
        return out;
    }
    public static long[] windowMax(long[] a, int k) {
        require(k > 0 && k <= a.length, "require 1 <= k <= n");
        int[] q = new int[k];
        int head = 0, count = 0;
        long[] out = new long[a.length - k + 1];
        for (int i = 0; i < a.length; i++) {
            if (count > 0 && q[head] <= i - k) {
                head = (head + 1) % k;
                count--;
            }
            while (count > 0 && a[q[(int)(((long)head + count - 1) % k)]] <= a[i])
                count--;
            q[(int)(((long)head + count) % k)] = i;
            count++;
            if (i >= k - 1)
                out[i - k + 1] = a[q[head]];
        }
        return out;
    }
    public static long[] mergeSort(long[] input) {
        long[] a = input.clone(), tmp = new long[a.length];
        int n = a.length;
        for (long width = 1; width < n; width *= 2) {
            for (long l = 0; l < n; l += 2 * width) {
                int lo = (int)l, mid = (int)Math.min(l + width, n),
                    hi = (int)Math.min(l + 2 * width, n);
                int i = lo, j = mid;
                for (int o = lo; o < hi; o++)
                    tmp[o] = i < mid && (j >= hi || a[i] <= a[j]) ? a[i++] : a[j++];
            }
            long[] swap = a;
            a = tmp;
            tmp = swap;
        }
        return a;
    }
    public static int lowerBound(long[] a, long target) {
        int lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (a[mid] < target)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }
    private static void siftDown(long[] heap, int size, int p, long value) {
        while (p < size / 2) {
            int c = 2 * p + 1;
            if (c + 1 < size && heap[c + 1] < heap[c])
                c++;
            if (heap[c] >= value)
                break;
            heap[p] = heap[c];
            p = c;
        }
        heap[p] = value;
    }
    public static long[] topKHeap(long[] a, int k) {
        require(k >= 0, "negative k");
        k = Math.min(k, a.length);
        long[] h = Arrays.copyOf(a, k);
        if (k == 0)
            return h;
        if (k < a.length) {
            for (int i = k / 2 - 1; i >= 0; i--)
                siftDown(h, k, i, h[i]);
            for (int i = k; i < a.length; i++)
                if (a[i] > h[0])
                    siftDown(h, k, 0, a[i]);
        }
        Arrays.sort(h);
        for (int i = 0, j = h.length - 1; i < j; i++, j--) {
            long v = h[i];
            h[i] = h[j];
            h[j] = v;
        }
        return h;
    }
    public static long[] topK(long[] a, int k) {
        require(k >= 0, "negative k");
        k = Math.min(k, a.length);
        if (k == 0)
            return new long[0];
        if ((long)k * 4 >= a.length) {
            long[] copy = a.clone();
            Arrays.sort(copy);
            long[] out = new long[k];
            for (int i = 0; i < k; i++)
                out[i] = copy[copy.length - 1 - i];
            return out;
        }
        return topKHeap(a, k);
    }
    public static final class TreeNode {
        public final long value;
        public TreeNode left, right;
        public TreeNode(long value) {
            this.value = value;
        }
    }
    private record Frame(TreeNode node, int phase) {}
    public static List<List<Long>> treeOrders(TreeNode root) {
        List<List<Long>> out = new ArrayList<>();
        for (int i = 0; i < 3; i++)
            out.add(new ArrayList<>());
        Deque<Frame> stack = new ArrayDeque<>();
        if (root != null)
            stack.push(new Frame(root, 0));
        while (!stack.isEmpty()) {
            Frame f = stack.pop();
            out.get(f.phase()).add(f.node().value);
            if (f.phase() < 2) {
                stack.push(new Frame(f.node(), f.phase() + 1));
                TreeNode child = f.phase() == 0 ? f.node().left : f.node().right;
                if (child != null)
                    stack.push(new Frame(child, 0));
            }
        }
        return out;
    }
    public static List<List<Long>> levelOrder(TreeNode root) {
        List<List<Long>> out = new ArrayList<>();
        Deque<TreeNode> q = new ArrayDeque<>();
        if (root != null)
            q.add(root);
        while (!q.isEmpty()) {
            int n = q.size();
            List<Long> row = new ArrayList<>(n);
            while (n-- > 0) {
                TreeNode u = q.remove();
                row.add(u.value);
                if (u.left != null)
                    q.add(u.left);
                if (u.right != null)
                    q.add(u.right);
            }
            out.add(row);
        }
        return out;
    }
    // Two-pass adjacency build keeps vertex arrays primitive, with deterministic edge order.
    private static int[][] graph(int n, int[][] edges, boolean undirected) {
        require(n >= 0, "negative vertex count");
        int[] count = new int[n];
        for (int[] e : edges) {
            require(e.length == 2 && e[0] >= 0 && e[0] < n && e[1] >= 0 && e[1] < n,
                    "invalid edge");
            count[e[0]]++;
            if (undirected)
                count[e[1]]++;
        }
        int[][] g = new int[n][];
        for (int i = 0; i < n; i++)
            g[i] = new int[count[i]];
        Arrays.fill(count, 0);
        for (int[] e : edges) {
            g[e[0]][count[e[0]]++] = e[1];
            if (undirected)
                g[e[1]][count[e[1]]++] = e[0];
        }
        return g;
    }
    public static boolean isBipartite(int n, int[][] edges) {
        int[][] g = graph(n, edges, true);
        int[] colors = new int[n], q = new int[n];
        Arrays.fill(colors, -1);
        for (int s = 0; s < n; s++) {
            if (colors[s] >= 0)
                continue;
            int head = 0, tail = 0;
            q[tail++] = s;
            colors[s] = 0;
            while (head < tail) {
                int u = q[head++];
                for (int v : g[u]) {
                    if (colors[v] < 0) {
                        colors[v] = colors[u] ^ 1;
                        q[tail++] = v;
                    } else if (colors[v] == colors[u])
                        return false;
                }
            }
        }
        return true;
    }
    public static int[] topologicalSort(int n, int[][] edges) {
        int[][] g = graph(n, edges, false);
        int[] degree = new int[n], order = new int[n];
        for (int[] e : edges)
            degree[e[1]]++;
        int tail = 0;
        for (int i = 0; i < n; i++)
            if (degree[i] == 0)
                order[tail++] = i;
        for (int head = 0; head < tail; head++)
            for (int v : g[order[head]])
                if (--degree[v] == 0)
                    order[tail++] = v;
        return tail == n ? order : null;
    }
    public record WeightedEdge(int from, int to, long weight) {}
    private record Distance(long distance, int vertex) implements Comparable<Distance> {
        public int compareTo(Distance b) {
            int c = Long.compare(distance, b.distance);
            return c != 0 ? c : Integer.compare(vertex, b.vertex);
        }
    }
    public static long[] dijkstra(int n, WeightedEdge[] edges, int source) {
        require(n > 0 && source >= 0 && source < n, "invalid source");
        int[] counts = new int[n];
        for (WeightedEdge e : edges) {
            require(e.from >= 0 && e.from < n && e.to >= 0 && e.to < n && e.weight >= 0 &&
                        e.weight <= DIST_LIMIT / n,
                    "invalid edge or unsafe weight");
            counts[e.from]++;
        }
        int[][] to = new int[n][];
        long[][] weight = new long[n][];
        for (int i = 0; i < n; i++) {
            to[i] = new int[counts[i]];
            weight[i] = new long[counts[i]];
        }
        Arrays.fill(counts, 0);
        for (WeightedEdge e : edges) {
            int p = counts[e.from]++;
            to[e.from][p] = e.to;
            weight[e.from][p] = e.weight;
        }
        long[] dist = new long[n];
        Arrays.fill(dist, INF);
        dist[source] = 0;
        PriorityQueue<Distance> q = new PriorityQueue<>();
        q.add(new Distance(0, source));
        while (!q.isEmpty()) {
            Distance d = q.remove();
            int u = d.vertex;
            if (d.distance != dist[u])
                continue;
            for (int i = 0; i < to[u].length; i++) {
                int v = to[u][i];
                long next = d.distance + weight[u][i];
                if (next < dist[v]) {
                    dist[v] = next;
                    q.add(new Distance(next, v));
                }
            }
        }
        return dist;
    }
    public static int[] prefixFunction(long[] p) {
        int[] pi = new int[p.length];
        for (int i = 1; i < p.length; i++) {
            int j = pi[i - 1];
            while (j > 0 && p[i] != p[j])
                j = pi[j - 1];
            if (p[i] == p[j])
                j++;
            pi[i] = j;
        }
        return pi;
    }
    public static int kmpSearch(long[] text, long[] p) {
        if (p.length == 0)
            return 0;
        int[] pi = prefixFunction(p);
        int j = 0;
        for (int i = 0; i < text.length; i++) {
            while (j > 0 && text[i] != p[j])
                j = pi[j - 1];
            if (text[i] == p[j])
                j++;
            if (j == p.length)
                return i - p.length + 1;
        }
        return -1;
    }
    public static final class Trie {
        private static final class Node {
            final Map<Long, Integer> children = new HashMap<>();
            boolean end;
        }
        private final List<Node> nodes = new ArrayList<>();
        public Trie() {
            nodes.add(new Node());
        }
        public void insert(long[] word) {
            int p = 0;
            for (long token : word) {
                Integer next = nodes.get(p).children.get(token);
                if (next == null) {
                    next = nodes.size();
                    nodes.get(p).children.put(token, next);
                    nodes.add(new Node());
                }
                p = next;
            }
            nodes.get(p).end = true;
        }
        private int find(long[] word) {
            int p = 0;
            for (long token : word) {
                Integer next = nodes.get(p).children.get(token);
                if (next == null)
                    return -1;
                p = next;
            }
            return p;
        }
        public boolean contains(long[] word) {
            int p = find(word);
            return p >= 0 && nodes.get(p).end;
        }
        public boolean startsWith(long[] prefix) {
            return find(prefix) >= 0;
        }
    }
    public record Point(long x, long y) {}
    private static void checkPoint(Point p) {
        require(p.x >= -1000000 && p.x <= 1000000 && p.y >= -1000000 && p.y <= 1000000,
                "geometry domain exceeded");
    }
    private static void checkPolygon(Point[] p) {
        require(p.length <= 100000, "too many vertices");
        for (Point q : p)
            checkPoint(q);
    }
    public static long polygonArea2(Point[] p) {
        checkPolygon(p);
        if (p.length < 3)
            return 0;
        long s = 0;
        for (int i = 0; i < p.length; i++) {
            Point a = p[i], b = p[(i + 1) % p.length];
            s += a.x * b.y - a.y * b.x;
        }
        return Math.abs(s);
    }
    public static int pointInPolygon(Point q, Point[] poly) {
        checkPolygon(poly);
        checkPoint(q);
        require(poly.length >= 3, "polygon needs three vertices");
        boolean inside = false;
        for (int i = 0; i < poly.length; i++) {
            Point a = poly[i], b = poly[(i + 1) % poly.length];
            long cross = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x);
            if (cross == 0 && q.x >= Math.min(a.x, b.x) && q.x <= Math.max(a.x, b.x) &&
                q.y >= Math.min(a.y, b.y) && q.y <= Math.max(a.y, b.y))
                return 0;
            if ((a.y > q.y) != (b.y > q.y) && ((cross > 0) == (b.y > a.y)))
                inside = !inside;
        }
        return inside ? 1 : -1;
    }
    public static List<int[]> combinations(int n, int k) {
        require(n >= 0 && k >= 0, "negative n or k");
        List<int[]> out = new ArrayList<>();
        if (k > n)
            return out;
        if (k == 0) {
            out.add(new int[0]);
            return out;
        }
        int[] row = new int[k];
        for (int i = 0; i < k; i++)
            row[i] = i + 1;
        while (true) {
            out.add(row.clone());
            int i = k - 1;
            while (i >= 0 && row[i] == n - k + i + 1)
                i--;
            if (i < 0)
                return out;
            row[i]++;
            for (int j = i + 1; j < k; j++)
                row[j] = row[j - 1] + 1;
        }
    }
    public static int editDistance(long[] a, long[] b) {
        if (a.length < b.length) {
            long[] t = a;
            a = b;
            b = t;
        }
        int[] row = new int[b.length + 1];
        for (int j = 0; j <= b.length; j++)
            row[j] = j;
        for (int i = 1; i <= a.length; i++) {
            int diagonal = row[0];
            row[0] = i;
            for (int j = 1; j <= b.length; j++) {
                int above = row[j];
                row[j] = Math.min(Math.min(above + 1, row[j - 1] + 1),
                                  diagonal + (a[i - 1] == b[j - 1] ? 0 : 1));
                diagonal = above;
            }
        }
        return row[b.length];
    }
    public static List<int[]> combinationsBacktrack(int n, int k) {
        require(n >= 0 && k >= 0, "negative n or k");
        List<int[]> out = new ArrayList<>();
        if (k > n)
            return out;
        require(k <= 128, "use iterative combinations for k > 128");
        combineVisit(n, k, 1, new int[k], 0, out);
        return out;
    }
    private static void combineVisit(int n, int k, long start, int[] path, int depth,
                                     List<int[]> out) {
        int need = k - depth;
        if (need == 0) {
            out.add(path.clone());
            return;
        }
        for (long value = start; value <= (long)n - need + 1; value++) {
            path[depth] = (int)value;
            combineVisit(n, k, value + 1, path, depth + 1, out);
        }
    }
    public static double sqrtNewton(double x) {
        require(Double.isFinite(x) && x >= 0, "require finite non-negative x");
        if (x == 0)
            return x;
        // Normalize subnormals first; Math.getExponent otherwise returns -1023 for all of them.
        int adjust = 0;
        double scaled = x;
        if (x < Double.MIN_NORMAL) {
            scaled = Math.scalb(x, 54);
            adjust = -54;
        }
        int e = Math.getExponent(scaled) + 1 + adjust;
        double m = Math.scalb(scaled, -(Math.getExponent(scaled) + 1));
        if (e % 2 != 0) {
            m *= 2;
            e--;
        }
        double t = 1;
        for (int i = 0; i < 16; i++) {
            double next = .5 * (t + m / t);
            if (Math.abs(next - t) <= 2 * Math.ulp(1.0) * next)
                return Math.scalb(next, e / 2);
            t = next;
        }
        throw new ArithmeticException("Newton iteration did not converge");
    }
    public static double[] gaussianSolve(double[][] a, double[] b) {
        int n = a.length;
        require(b.length == n, "shape mismatch");
        double[][] aug = new double[n][n + 1];
        for (int r = 0; r < n; r++) {
            require(a[r].length == n && Double.isFinite(b[r]), "shape or nonfinite input");
            double scale = 0;
            for (double v : a[r]) {
                require(Double.isFinite(v), "nonfinite input");
                scale = Math.max(scale, Math.abs(v));
            }
            require(scale > 0, "numerically singular");
            for (int c = 0; c < n; c++)
                aug[r][c] = a[r][c] / scale;
            aug[r][n] = b[r] / scale;
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++)
                if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col]))
                    pivot = r;
            require(Math.abs(aug[pivot][col]) > 1e-12, "numerically singular");
            double[] t = aug[col];
            aug[col] = aug[pivot];
            aug[pivot] = t;
            for (int r = col + 1; r < n; r++) {
                double f = aug[r][col] / aug[col][col];
                aug[r][col] = 0;
                for (int c = col + 1; c <= n; c++)
                    aug[r][c] -= f * aug[col][c];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = 0;
            for (int j = i + 1; j < n; j++)
                s += aug[i][j] * x[j];
            x[i] = (aug[i][n] - s) / aug[i][i];
            require(Double.isFinite(x[i]), "nonfinite result");
        }
        return x;
    }
    public static final class DSU {
        private final int[] parent, size;
        public DSU(int n) {
            require(n >= 0, "negative n");
            parent = new int[n];
            size = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                size[i] = 1;
            }
        }
        public int find(int x) {
            require(x >= 0 && x < parent.length, "index out of range");
            while (x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }
        public boolean union(int a, int b) {
            a = find(a);
            b = find(b);
            if (a == b)
                return false;
            if (size[a] < size[b]) {
                int t = a;
                a = b;
                b = t;
            }
            parent[b] = a;
            size[a] += size[b];
            return true;
        }
    }
    public static final class Fenwick {
        private final long[] tree;
        public Fenwick(int n) {
            require(n >= 0 && n < Integer.MAX_VALUE, "invalid n");
            tree = new long[n + 1];
        }
        public void add(int index, long delta) {
            require(index >= 0 && index < tree.length - 1, "index out of range");
            for (long i = (long)index + 1; i < tree.length; i += i & -i)
                tree[(int)i] += delta;
        }
        public long prefixSum(int end) {
            require(end >= 0 && end < tree.length, "end out of range");
            long s = 0;
            while (end > 0) {
                s += tree[end];
                end -= end & -end;
            }
            return s;
        }
    }
}
