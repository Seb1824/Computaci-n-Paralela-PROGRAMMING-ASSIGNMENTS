#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int local_val, sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_val = rank + 1;
    sum = local_val;

    int steps = log2(size);
    for (int i = 0; i < steps; i++) {
        int partner = rank ^ (1 << i);
        int recv_val;
        MPI_Sendrecv(&sum, 1, MPI_INT, partner, 0,
                     &recv_val, 1, MPI_INT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sum += recv_val;
    }

    printf("Proceso %d: suma global = %d\n", rank, sum);

    MPI_Finalize();
    return 0;
}
