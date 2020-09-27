import java.util.Scanner;

public class OJ3_P1 {

    // Print the given sequence
    public static void print_seq(int [] seq) {
        for (int i=0;i<seq.length;i++){
            System.out.print(seq[i]);
            if (i != seq.length -1){
                System.out.print(" ");
            }
        }
    }

    // Main
    public static void main(String [] args) {
        Scanner kb = new Scanner(System.in);
        String [] input = kb.nextLine().split(" ");
        int n = Integer.parseInt(input[0]);
        int k = Integer.parseInt(input[1]);

        int [] seq = new int[n];
        int [] dirs = new int[n];

        // Create the initial sequence
        for (int i=1;i<n+1;i++){
            seq[i-1] = i;
        }

    }
}
