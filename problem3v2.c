#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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
    } else {
        if (rank + p < size) {
            int recv_val;
            MPI_Recv(&recv_val, 1, MPI_INT, rank + p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += recv_val;
        }

        int step = 1;
        while (step < p) {
            if (rank % (2 * step) == 0) {
                int recv_val;
                MPI_Recv(&recv_val, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sum += recv_val;
            } else if (rank % step == 0) {
                MPI_Send(&sum, 1, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
                break;
            }
            step *= 2;
        }

        if (rank == 0) {
            printf("Suma global (general): %d\n", sum);
        }
    }

    MPI_Finalize();
    return 0;
}

