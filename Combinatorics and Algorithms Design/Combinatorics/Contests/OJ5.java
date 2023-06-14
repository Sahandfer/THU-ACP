import java.util.*;

class Main {
    public static int[][] entries= new int[1001][1001];
    public static int x1, x2, y1, y2, dir_x, dir_y;

    public static int find_lattice_path(int x, int y){
        // Invalid
        if (x<0 || x>1000 || y<0 || y>1000) return 0;
        if ((x > x1 && x > x2) || (y > y1 && y > y2)|| (x < x1 && x < x2) || (y < y1 && y < y2)) return 0;

        // If already recorded
        if(entries[x][y]!=-6) return entries[x][y];

        entries[x][y]=
                (int) (
                        find_lattice_path(x+dir_x,y)%(Math.pow(10,9)+7)+
                        find_lattice_path(x,y+dir_y)%(Math.pow(10,9)+7)
                );

        return entries[x][y];
    }

    public static void main(String [] args) {
        Scanner kb = new Scanner(System.in);
        String [] input = kb.nextLine().split(" ");
        x1 = Integer.parseInt(input[0])+500;
        y1 = Integer.parseInt(input[1])+500;
        x2 = Integer.parseInt(input[2])+500;
        y2 = Integer.parseInt(input[3])+500;


        dir_x = x2 > x1 ? 1 : -1;
        dir_y = y2 > y1 ? 1 : -1;

        // Fill the array with non-reachable number -6 (so that we know which values have been recorded before)
        for (int i =0;i<1001;i++) {
            for (int j=0; j<1001; j++) {
                if (i!=j) entries[i][j] = -6;
            }
        }

        entries[x1][y1] = -6; // starting point
        entries[x2][y2] = 1; // destination

        System.out.println((int)(find_lattice_path(x1, y1)%(Math.pow(10,9)+7)));
    }

}