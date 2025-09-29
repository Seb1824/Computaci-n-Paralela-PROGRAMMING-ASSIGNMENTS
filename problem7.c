#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int ping_pongs = 100000;
    int msg = 1;
    clock_t start_c, end_c;
    double start_mpi, end_mpi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Se necesitan al menos 2 procesos.\n");
        MPI_Finalize();
        return 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        start_c = clock();
        start_mpi = MPI_Wtime();
        for (int i = 0; i < ping_pongs; i++) {
            MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        end_c = clock();
        end_mpi = MPI_Wtime();
        double time_c = ((double)(end_c - start_c)) / CLOCKS_PER_SEC;
        double time_mpi = end_mpi - start_mpi;
        printf("Tiempo con clock():   %f segundos\n", time_c);
        printf("Tiempo con MPI_Wtime: %f segundos\n", time_mpi);
        printf("Promedio por ping-pong (MPI_Wtime): %e segundos\n", time_mpi / ping_pongs);
    } else if (rank == 1) {
        for (int i = 0; i < ping_pongs; i++) {
            MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
