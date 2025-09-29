#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static void pack_block(double* A, int n, int r, int bi, int bj, double* dst) {
    for (int i = 0; i < r; i++) {
        memcpy(dst + i*r, A + (bi*r + i)*n + bj*r, r*sizeof(double));
    }
}

int main(int argc, char* argv[]) {
    int rank, size, n, q, coords[2], dims[2], periods[2] = {0,0};
    double *A = NULL, *x = NULL, *y = NULL;
    double *local_A, *local_x, *local_y, *y_block;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    q = (int)sqrt((double)size);
    if (q*q != size) { if (rank==0) printf("Error: comm_sz no es cuadrado perfecto.\n"); MPI_Finalize(); return 0; }

    if (rank == 0) {
        printf("Ingrese el orden de la matriz (n, divisible entre %d): ", q);
        fflush(stdout);
        if (scanf("%d", &n) != 1) { MPI_Abort(MPI_COMM_WORLD, 1); }
        if (n % q != 0) { printf("Error: n no es divisible entre %d.\n", q); MPI_Abort(MPI_COMM_WORLD, 1); }
        A = (double*)malloc((size_t)n*n*sizeof(double));
        x = (double*)malloc((size_t)n*sizeof(double));
        for (int i=0;i<n;i++){ x[i]=1.0; for (int j=0;j<n;j++) A[i*(size_t)n+j]=i+j+1; }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int r = n / q;

    MPI_Comm cart, row_comm, col_comm, diag_comm;
    dims[0]=q; dims[1]=q;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);
    MPI_Cart_coords(cart, rank, 2, coords);

    int row = coords[0], col = coords[1];
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);
    if (row==col) MPI_Comm_split(MPI_COMM_WORLD, 0, row, &diag_comm);
    else diag_comm = MPI_COMM_NULL;

    local_A = (double*)malloc((size_t)r*r*sizeof(double));
    local_x = (double*)malloc((size_t)r*sizeof(double));
    local_y = (double*)calloc((size_t)r, sizeof(double));
    y_block = (double*)malloc((size_t)r*sizeof(double));

    if (rank == 0) {
        double* buf = (double*)malloc((size_t)r*r*sizeof(double));
        for (int bi=0; bi<q; bi++) {
            for (int bj=0; bj<q; bj++) {
                int dest; int c[2]={bi,bj};
                MPI_Cart_rank(cart, c, &dest);
                pack_block(A, n, r, bi, bj, buf);
                if (dest == 0) memcpy(local_A, buf, (size_t)r*r*sizeof(double));
                else MPI_Send(buf, r*r, MPI_DOUBLE, dest, 100, MPI_COMM_WORLD);
            }
        }
        for (int k=0; k<q; k++) {
            int c[2]={k,k}; int dest; MPI_Cart_rank(cart, c, &dest);
            if (dest == 0) memcpy(local_x, x + k*r, (size_t)r*sizeof(double));
            else MPI_Send(x + k*r, r, MPI_DOUBLE, dest, 200, MPI_COMM_WORLD);
        }
        free(buf);
    } else {
        MPI_Recv(local_A, r*r, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (row==col) MPI_Recv(local_x, r, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int root_in_col = col;
    MPI_Bcast(local_x, r, MPI_DOUBLE, root_in_col, col_comm);

    for (int i=0;i<r;i++)
        for (int j=0;j<r;j++)
            local_y[i] += local_A[i*(size_t)r + j] * local_x[j];

    int root_in_row = row;
    if (row == col) MPI_Reduce(local_y, y_block, r, MPI_DOUBLE, MPI_SUM, root_in_row, row_comm);
    else MPI_Reduce(local_y, NULL, r, MPI_DOUBLE, MPI_SUM, root_in_row, row_comm);

    if (diag_comm != MPI_COMM_NULL) {
        if (row == 0) y = (double*)malloc((size_t)n*sizeof(double));
        MPI_Gather(y_block, r, MPI_DOUBLE, y, r, MPI_DOUBLE, 0, diag_comm);
    }

    if (rank == 0) {
        printf("Resultado y = A*x:\n");
        for (int i=0;i<n;i++) printf("%5.1f\n", y[i]);
    }

    free(local_A); free(local_x); free(local_y); free(y_block);
    if (rank==0){ free(A); free(x); free(y); }
    if (diag_comm != MPI_COMM_NULL) MPI_Comm_free(&diag_comm);
    MPI_Comm_free(&row_comm); MPI_Comm_free(&col_comm); MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
