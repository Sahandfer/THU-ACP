import java.util.*;

public class OJ3_P2 {

    static long[] fac = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800L, 87178291200L, 1307674368000L, 20922789888000L, 355687428096000L, 6402373705728000L, 121645100408832000L, 2432902008176640000L};

    public static List<Integer> toList(int[] nums) {
        List<Integer> res = new LinkedList<>();
        for (int num : nums) {
            res.add(num);
        }
        return res;
    }

    public static List<Integer> permutation(int[] nums, long k) {
        List<Integer> source = toList(nums);
        List<Integer> result = new ArrayList(nums.length);
        while (!source.isEmpty()) {
            long f = fac[source.size() - 1];
            result.add(source.remove( (int)(k / f)));
            k %= f;
        }
        return result;
    }

    public static long reverse_perm(int [] nums, List<Integer> perm) {
        long k=0;
        int idx = nums.length-1;
        List<Integer> used = new ArrayList<>();

        for (int item: perm) {
            int acc = item -1;
            for (int i=0;i<item -1; i++) {
                if (used.contains(nums[i])){
                    acc--;
                }
            }
            used.add(item);
            k = k + (acc*fac[idx]);
            idx--;
        }
        return k;
    }


    public static void main(String [] args) {
        Scanner kb = new Scanner(System.in);
        String [] input = kb.nextLine().split(" ");
        int k = Integer.parseInt(input[0]);
        int a = Integer.parseInt(input[1]);
        int [] seq = new int[k];
        List<Integer> k_perm =  new ArrayList<>();

        for (int i=0; i< k; i++) {
            seq[i] = i+1;
            k_perm.add(kb.nextInt());
        }

        long kth = reverse_perm(seq, k_perm);
        List<Integer> res = permutation(seq, kth+a);
        for (int item:res) {
            System.out.print(item);
            System.out.print(" ");
        }
        System.out.println();

    }
}
