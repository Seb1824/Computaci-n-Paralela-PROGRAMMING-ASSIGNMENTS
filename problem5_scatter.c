#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size, n;
    double *A = NULL, *x = NULL;
    double *local_A, *local_x, *local_y, *y;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Ingrese el orden de la matriz (n, divisible entre %d): ", size);
        fflush(stdout);
        scanf("%d", &n);

        A = (double*) malloc(n * n * sizeof(double));
        x = (double*) malloc(n * sizeof(double));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i*n + j] = i + j + 1;
            }
        }
        for (int i = 0; i < n; i++) {
            x[i] = 1.0;
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_cols = n / size;
    local_A = (double*) malloc(n * local_cols * sizeof(double));
    local_x = (double*) malloc(local_cols * sizeof(double));
    local_y = (double*) calloc(n, sizeof(double));
    y = (double*) malloc((n/size) * sizeof(double));

    MPI_Scatter(A, n * local_cols, MPI_DOUBLE,
                local_A, n * local_cols, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Scatter(x, local_cols, MPI_DOUBLE,
                local_x, local_cols, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < local_cols; j++) {
            local_y[i] += local_A[i*local_cols + j] * local_x[j];
        }
    }

    int recvcounts[size];
    for (int i = 0; i < size; i++) recvcounts[i] = n/size;

    MPI_Reduce_scatter(local_y, y, recvcounts, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < n/size; i++) {
        printf("Proceso %d: y[%d] = %5.1f\n", rank, rank*(n/size)+i, y[i]);
    }

    free(local_A);
    free(local_x);
    free(local_y);
    free(y);
    if (rank == 0) {
        free(A);
        free(x);
    }

    MPI_Finalize();
    return 0;
}
