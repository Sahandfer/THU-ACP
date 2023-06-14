import java.util.*;
import java.util.HashMap;

class Main {
    public static long[][] entries= new long[2001][2001];
    public static long find_Integer_partitions(int n, int k) {
        if (k == 0 || n < 0) return 0;
        if (n == 0 || k == 1) return 1;
        if (entries[n][k] != 0) return entries[n][k];
        entries[n][k] = (long) ((find_Integer_partitions(n, k - 1) + find_Integer_partitions(n - k, k))%(Math.pow(10,9)+7));
        return entries[n][k];
    }

    public static void main(String [] args) {
        Scanner kb = new Scanner(System.in);
        String [] input = kb.nextLine().split(" ");
        int n = Integer.parseInt(input[0]);
        int k = Integer.parseInt(input[1]);
        System.out.println((int)(find_Integer_partitions(n,k)%(Math.pow(10,9)+7)));
//        System.out.println((int)(find_Integer_partitions(1957, 1844)%(Math.pow(10,9)+7))); //643871523
    }

}
