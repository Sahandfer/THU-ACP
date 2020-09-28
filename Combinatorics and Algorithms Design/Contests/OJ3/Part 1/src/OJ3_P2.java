import java.util.Arrays;
import java.util.Scanner;

public class OJ3_P2 {
    public static void find_next_perm(int[] seq, int a){
        for (int i=0; i<a; i++) {
            for (int j=seq.length-1; j>0; j--) {
                if (seq[j]>seq[j-1]) {
                    Arrays.sort(seq, j-1, seq.length);
                    int temp = seq[j];
                    seq[j] = seq[j-1];
                    seq[j-1] = temp;
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
        System.out.println();
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
