#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size, n;
    double *A = NULL, *x = NULL, *y = NULL;
    double *local_A, *local_x, *local_y;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Ingrese el orden de la matriz (n, divisible entre %d): ", size);
        fflush(stdout);
        scanf("%d", &n);

        A = (double*) malloc(n * n * sizeof(double));
        x = (double*) malloc(n * sizeof(double));

        printf("Matriz A (%dx%d):\n", n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i*n + j] = i + j + 1;
                printf("%5.1f ", A[i*n + j]);
            }
            printf("\n");
        }

        printf("Vector x:\n");
        for (int i = 0; i < n; i++) {
            x[i] = 1.0;
            printf("%5.1f\n", x[i]);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_cols = n / size;
    local_A = (double*) malloc(n * local_cols * sizeof(double));
    local_x = (double*) malloc(local_cols * sizeof(double));
    local_y = (double*) calloc(n, sizeof(double));

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

    if (rank == 0) {
        y = (double*) malloc(n * sizeof(double));
    }
    MPI_Reduce(local_y, y, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Resultado y = A*x:\n");
        for (int i = 0; i < n; i++) {
            printf("%5.1f\n", y[i]);
        }
        free(A);
        free(x);
        free(y);
    }

    free(local_A);
    free(local_x);
    free(local_y);

    MPI_Finalize();
    return 0;
}
