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

    int p = 1;
    while (p * 2 <= size) {
        p *= 2;
    }

    if (rank >= p) {
        MPI_Send(&sum, 1, MPI_INT, rank - p, 0, MPI_COMM_WORLD);
        MPI_Recv(&sum, 1, MPI_INT, rank - p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        if (rank + p < size) {
            int extra;
            MPI_Recv(&extra, 1, MPI_INT, rank + p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += extra;
        }

        int steps = log2(p);
        for (int i = 0; i < steps; i++) {
            int partner = rank ^ (1 << i);
            int recv_val;
            MPI_Sendrecv(&sum, 1, MPI_INT, partner, 0,
                         &recv_val, 1, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += recv_val;
        }

        if (rank + p < size) {
            MPI_Send(&sum, 1, MPI_INT, rank + p, 0, MPI_COMM_WORLD);
        }
    }

    printf("Proceso %d: suma global = %d\n", rank, sum);

    MPI_Finalize();
    return 0;
}

