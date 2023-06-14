#include "your_reduce.h"
#include <cmath>
#include <omp.h>

#define MAX_LEN 268435456
#define NUM_THREADS 4
// You may add your functions and variables here

// Function for adding elements of the two buffers one by one
void Sum_Buffers(const int *buffer_a, int *buffer_b, int *recvbuf, int count)
{
#pragma omp parallel for num_threads(NUM_THREADS) schedule(guided)
    for (int i = 0; i < count; i++)
        recvbuf[i] += buffer_a[i] + buffer_b[i];
}

void YOUR_Reduce(const int *sendbuf, int *recvbuf, int count)
{
    /*
        Modify the code here.
        Your implementation should have the same result as this MPI_Reduce
        call. However, you MUST NOT use MPI_Reduce (or like) for your hand-in
        version. Instead, you should use MPI_Send and MPI_Recv (or like). See
        the homework instructions for more information.
    */
    // MPI_Reduce(sendbuf, recvbuf, count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    /*
        You may assume:
        - Data type is always `int` (MPI_INT).
        - Operation is always MPI_SUM.
        - Process to hold final results is always process 0.
        - Number of processes is 2, 4, or 8.
        - Number of elements (`count`) is 1, 16, 256, 4096, 65536, 1048576,
          16777216, or 268435456.
        For other cases, your code is allowed to produce wrong results or even
        crash. It is totally fine.
    */
    int dest, source;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Clear the receive buffer for each node
    for (int i = 0; i < count; i++)
        recvbuf[i] = 0;

    if (rank % 2 != 0)
    {
        dest = rank - 1;
        MPI_Send(sendbuf, count, MPI_INT, dest, 0, MPI_COMM_WORLD);
    }
    else
    {
        // Initialize array to hold the received buffer
        int *tempbuff;
        tempbuff = (int *)malloc(MAX_LEN * sizeof(int));
        // Receive buffer from other ranks
        source = rank + 1;
        MPI_Recv(tempbuff, count, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Sum the received buffer with the current buffer
        Sum_Buffers(sendbuf, tempbuff, recvbuf, count);
        // This step is only necessary for more than 2 nodes
        if (size > 2)
        {
            // For nodes 2 and 6
            if (rank % 4 != 0)
            {
                dest = (rank / 4) * 4; // 2 -> 0 and 6 -> 4
                MPI_Send(recvbuf, count, MPI_INT, dest, 0, MPI_COMM_WORLD);
            }
            else
            {
                // For node 4
                if (rank != 0)
                {
                    // Receive the buffer from node 6 and sum the results
                    MPI_Recv(tempbuff, count, MPI_INT, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    int *temp = (int *)malloc(MAX_LEN * sizeof(int));
                    Sum_Buffers(temp, tempbuff, recvbuf, count);
                    // Send the results to node 0 for final sum
                    MPI_Send(recvbuf, count, MPI_INT, 0, 0, MPI_COMM_WORLD);
                }
                // For node 0
                else
                {
                    // Receive results from nodes 2 (if size = 4 or 8) and 4 (if size = 8) and calculate sum
                    for (int j = 1; j < log2(size); j++)
                    {
                        source = j * 2;
                        MPI_Recv(tempbuff, count, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        int *temp = (int *)malloc(MAX_LEN * sizeof(int));
                        Sum_Buffers(temp, tempbuff, recvbuf, count);
                    }
                }
            }
        }
    }
}
