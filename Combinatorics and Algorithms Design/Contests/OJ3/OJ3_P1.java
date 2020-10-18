import java.lang.reflect.Array;
import java.util.*;

public class OJ3_P1 {

    public static void find_kth_perm(int n, long k) {
        ArrayList<Integer> seq = new ArrayList<>(n);
        long[] fac = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800L, 87178291200L, 1307674368000L, 20922789888000L, 355687428096000L, 6402373705728000L, 121645100408832000L, 2432902008176640000L};
        long [] new_pos = new long[n];
        int [] num_changes = new int[n];
        seq.add(1);
        for (int i=1; i<n+1;i++) {
            new_pos[i-1] = -6;
            num_changes[i-1] = 0;
        }

        long fact = fac[n];
        k--;

        // See how many times each permutation has changed
        for (int i=2; i<n+1;i++) {
            fact/=i;
            long num_change = k/fact;
            if (num_change != 0) {
                long gp = num_change/i;
                long remain = num_change%i;
                if (gp%2 == 0) {
                    new_pos[i-1] = (i-1-remain);
                }
                else {
                    new_pos[i-1] = remain;
                }
                seq.add((int) new_pos[i-1], i);
            }
            else{
                seq.add(i);
            }
        }

        print_perm(seq);

    }

    public static void print_perm(ArrayList<Integer> perm) {
        for (int digit: perm) {
            System.out.print(digit);
            System.out.print(" ");
        }
        System.out.println();
    }

    public static void print_seq(int [] seq) {
        for (int i=0;i<seq.length;i++){
            System.out.print(seq[i]);
            if (i != seq.length -1){
                System.out.print(" ");
            }
        }
        System.out.println();
    }

    public static void main(String [] args) {
        Scanner kb = new Scanner(System.in);
        String [] input = kb.nextLine().split(" ");
        int n = Integer.parseInt(input[0]);
        long k = Long.parseLong(input[1]);
//        int k = 11;

        int [] seq = new int[n];

        for (int i=0; i< n; i++) {
            seq[i] = i+1;
        }

        if (k==1) print_seq(seq);
        else find_kth_perm(n, k);


    }
}
