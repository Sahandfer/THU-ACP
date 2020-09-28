import java.util.Scanner;
class OJ3_P1 {

    // Find largest mobile number
    public static int find_largest_mobile(int [] seq, int [] dirs, int n) {
        int max_mobile = 0;
        int max_mobile_idx = 0;

        for (int i=0; i<n; i++) {
            if(dirs[seq[i]-1] != 0 && seq[i] > max_mobile) {
                max_mobile = seq[i];
                max_mobile_idx = i;
                if (max_mobile == n) break;
            }
        }

        return max_mobile_idx;
    }

    // Print the given sequence
    public static void print_seq(int [] seq) {
        for (int i=0;i<seq.length;i++){
            System.out.print(seq[i]);
            if (i != seq.length -1){
                System.out.print(" ");
            }
        }
        System.out.println();
    }

    // Find the nth permutation
    public static void find_kth_perm(int n, long k) {
        int [] seq = new int[n];
        int [] dirs = new int[n];

        // Create the initial sequence and directions
        for (int i=0;i<n;i++){
            seq[i] = i+1;
            dirs[i] = i == 0 ? 0 : -1; // 0 for not moving, 1 for right and -1 for left
        }

        for (int i=0; i<k-1;i++) {
            int mobile_idx = find_largest_mobile(seq, dirs, n);
            int mobile_val = seq[mobile_idx];
            int adj_idx = mobile_idx + dirs[seq[mobile_idx]-1];

            seq[mobile_idx] = seq[adj_idx];
            seq[adj_idx] = mobile_val;

            int adj_idx2 = adj_idx + dirs[seq[adj_idx]-1];

            if (adj_idx == 0 || adj_idx == n-1 || seq[adj_idx2]>seq[adj_idx]) {
                dirs[seq[adj_idx]-1] = 0;
            }

            for (int j=0; j<n;j++) {
                if (seq[j] > mobile_val && dirs[seq[j] - 1] == 0) {
                    if (j < mobile_idx) {
                        dirs[seq[j] - 1] = 1;
                    } else if (j > mobile_idx) {
                        dirs[seq[j] - 1] = -1;
                    }
                }
            }

            print_seq(seq);
        }
    }

    // Main
    public static void main(String [] args) {
        Scanner kb = new Scanner(System.in);
//        String [] input = kb.nextLine().split(" ");
//        int n = Integer.parseInt(input[0]);
//        long k = Long.parseLong(input[1]);
        find_kth_perm(3, 6);
    }
}
