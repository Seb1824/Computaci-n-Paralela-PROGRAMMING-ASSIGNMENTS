#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]) {
    int rank, size, n, q;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    q = (int)sqrt((double)size);
    if (q*q != size) {
        if (rank == 0) printf("Error: comm_sz no es cuadrado perfecto.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("Ingrese el orden de la matriz (n, divisible entre %d): ", q);
        fflush(stdout);
        if (scanf("%d", &n) != 1) MPI_Abort(MPI_COMM_WORLD, 1);
        if (n % q != 0) {
            printf("Error: n no es divisible entre %d.\n", q);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int r = n / q;

    double* A = NULL;
    double* x = NULL;
    double* y = NULL;
    if (rank == 0) {
        A = (double*)malloc(n*n*sizeof(double));
        x = (double*)malloc(n*sizeof(double));
        y = (double*)malloc(n*sizeof(double));
        for (int i=0;i<n;i++) {
            x[i] = 1.0;
            for (int j=0;j<n;j++) {
                A[i*n+j] = i+j+1;
            }
        }
    }

    int rows = n / size;
    double* local_A = (double*)malloc(rows*n*sizeof(double));
    double* local_y = (double*)calloc(rows, sizeof(double));

    int* sendcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size*sizeof(int));
        displs = (int*)malloc(size*sizeof(int));
        for (int p=0;p<size;p++) {
            sendcounts[p] = rows*n;
            displs[p] = p*rows*n;
        }
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, rows*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) x = (double*)malloc(n*sizeof(double));
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i=0;i<rows;i++)
        for (int j=0;j<n;j++)
            local_y[i] += local_A[i*n+j] * x[j];

    MPI_Gather(local_y, rows, MPI_DOUBLE, y, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Resultado y = A*x:\n");
        for (int i=0;i<n;i++) printf("%5.1f\n", y[i]);
        free(A); free(x); free(y);
        free(sendcounts); free(displs);
    } else {
        free(x);
    }

    free(local_A); free(local_y);
    MPI_Finalize();
    return 0;
}
