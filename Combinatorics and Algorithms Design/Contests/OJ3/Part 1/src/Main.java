import java.util.*;

class Main {




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


    }
}
