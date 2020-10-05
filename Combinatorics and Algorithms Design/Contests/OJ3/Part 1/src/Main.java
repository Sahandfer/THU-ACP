import java.util.*;

class Main {

    static long[] fac = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800L, 87178291200L, 1307674368000L, 20922789888000L, 355687428096000L, 6402373705728000L, 121645100408832000L, 2432902008176640000L};

    public static List<Integer> perm(List<Integer> nums, long k) {
        List<Integer> perms = new ArrayList(nums.size());
        while (!nums.isEmpty()) {
            long f = fac[nums.size() - 1];
            perms.add(nums.remove((int)(k / f)));
            k %= f;
        }
        return perms;
    }

    public static long reverse_perm(List<Integer> nums, List<Integer> perm) {
        long k=0;
        int idx = nums.size()-1;
        List<Integer> used = new ArrayList<>();

        for (int item: perm) {
            int acc = item -1;
            for (int i=0;i<item -1; i++) {
                if (used.contains(nums.get(i))){
                    acc--;
                }
            }
            used.add(item);
            k = k + (acc*fac[idx]);
            idx--;
        }
        return k;
    }

    public static void print_perm(List<Integer> perm) {
        for (int item:perm) {
            System.out.print(item);
            System.out.print(" ");
        }
        System.out.println();
    }

    public static void find_kath_perm(int k, int a, Scanner kb) {
        List<Integer> seq_list = new ArrayList<>();
        List<Integer> k_perm =  new ArrayList<>();

        for (int i=0; i< k; i++) {
            seq_list.add(i+1);
            k_perm.add(kb.nextInt());
        }

        long kth = reverse_perm(seq_list, k_perm);
        List<Integer> kath = perm(seq_list, kth + a);
        print_perm(kath);
    }


    public static void main(String [] args) {
        Scanner kb = new Scanner(System.in);
        String [] input = kb.nextLine().split(" ");
        int k = Integer.parseInt(input[0]);
        int a = Integer.parseInt(input[1]);

        find_kath_perm(k, a, kb);
    }
}
