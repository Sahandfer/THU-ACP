import java.util.Arrays;
import java.util.Scanner;

class Main {
    public static int find_small (int[] seq, int start, int val) {
        int min_val_idx = seq.length-1;

        int min = start;
        int max = seq.length - 1;
        int res=0;

        while(true) {
            int avg_idx = Math.round((min+max)/2);
            int avg_val = seq[avg_idx];
            if (avg_val == val) {
                min_val_idx = avg_idx;
                break;
            }
        }


        for (int i=start;i< seq.length;i++) {
            if (seq[i] - val > 0 ) {
                min_val_idx = i;
                break;
            }
        }
        return min_val_idx;
    }
    public static void find_next_perm(int[] seq, int a){
        for (int i=0; i<a; i++) {
            for (int j=seq.length-1; j>0; j--) {
                if (seq[j]>seq[j-1]) {
                    Arrays.sort(seq, j, seq.length);
                    int min_val_idx = find_small(seq, j, seq[j-1]);
                    int temp = seq[j-1];
                    seq[j-1] = seq[min_val_idx];
                    seq[min_val_idx] = temp;
                    break;
                }
            }
        }

        for (int i=0; i<seq.length;i++){
            System.out.print(seq[i]);
            if (i != seq.length -1){
                System.out.print(" ");
            }
        }
    }

    public static void main(String [] args) {
        Scanner kb = new Scanner(System.in);
        String [] input = kb.nextLine().split(" ");
        int k = Integer.parseInt(input[0]);
        int a = Integer.parseInt(input[1]);
        int [] seq = new int[k];

        for (int i=0; i< k; i++) {
            seq[i] = kb.nextInt();
        }

        find_next_perm(seq, a);
    }
}
