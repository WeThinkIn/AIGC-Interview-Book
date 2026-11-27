import java.util.*;
import java.util.function.Supplier;
public final class Bench {
    static final int N = 32768;
    static volatile long sink;
    static long[] input() {
        long state = 20261127;
        long[] a = new long[N];
        for (int i = 0; i < N; i++) {
            state = (1664525 * state + 1013904223) & 0xffffffffL;
            a[i] = state % 100000;
        }
        return a;
    }
    static void measure(String task, String method, Supplier<long[]> fn, long[] expected) {
        if (!Arrays.equals(fn.get(), expected))
            throw new AssertionError("benchmark correctness");
        for (int i = 0; i < 40; i++) {
            long[] a = fn.get();
            sink = a.length == 0 ? 0 : a[0];
        }
        StringBuilder out = new StringBuilder("{\"task\":\"")
                                .append(task)
                                .append("\",\"method\":\"")
                                .append(method)
                                .append("\",\"samples_ms\":[");
        for (int i = 0; i < 11; i++) {
            long t = System.nanoTime();
            long[] a = fn.get();
            long end = System.nanoTime();
            sink = a.length == 0 ? 0 : a[0];
            if (i > 0)
                out.append(',');
            out.append((end - t) / 1e6);
        }
        System.out.println(out.append("],\"sink\":").append(sink).append('}'));
    }
    static long[] sorted(long[] a) {
        long[] b = a.clone();
        Arrays.sort(b);
        return b;
    }
    static long[] fullTop(long[] a, int k) {
        long[] b = sorted(a), out = new long[k];
        for (int i = 0; i < k; i++)
            out[i] = b[b.length - 1 - i];
        return out;
    }
    static long[] pqTop(long[] a, int k) {
        PriorityQueue<Long> q = new PriorityQueue<>();
        for (long x : a) {
            if (q.size() < k)
                q.add(x);
            else if (x > q.peek()) {
                q.remove();
                q.add(x);
            }
        }
        long[] out = new long[k];
        for (int i = k - 1; i >= 0; i--)
            out[i] = q.remove();
        return out;
    }
    static long[] scan(long[] a, int k) {
        long[] out = new long[a.length - k + 1];
        for (int i = 0; i < out.length; i++) {
            long m = a[i];
            for (int j = 1; j < k; j++)
                m = Math.max(m, a[i + j]);
            out[i] = m;
        }
        return out;
    }
    public static void main(String[] args) {
        long[] raw = input();
        for (String kind : List.of("random", "sorted", "reversed", "duplicates")) {
            long[] a = raw.clone();
            if (kind.equals("sorted") || kind.equals("reversed"))
                Arrays.sort(a);
            if (kind.equals("reversed")) {
                for (int i = 0, j = a.length - 1; i < j; i++, j--) {
                    long v = a[i];
                    a[i] = a[j];
                    a[j] = v;
                }
            }
            if (kind.equals("duplicates"))
                for (int i = 0; i < a.length; i++)
                    a[i] %= 8;
            long[] expected = sorted(a);
            measure("sort_" + kind, "merge_buffers", () -> Algorithms.mergeSort(a), expected);
            measure("sort_" + kind, "arrays_sort", () -> sorted(a), expected);
        }
        for (int k : new int[] {8, 128, N / 2}) {
            long[] expected = fullTop(raw, k);
            String task = "top_" + k;
            measure(task, "primitive_heap", () -> Algorithms.topKHeap(raw, k), expected);
            measure(task, "adaptive", () -> Algorithms.topK(raw, k), expected);
            measure(task, "priority_queue", () -> pqTop(raw, k), expected);
            measure(task, "full_sort", () -> fullTop(raw, k), expected);
        }
        long[] expected = scan(raw, 128);
        measure("window_128", "monotonic_ring", () -> Algorithms.windowMax(raw, 128), expected);
        measure("window_128", "scan_each_window", () -> scan(raw, 128), expected);
    }
}
