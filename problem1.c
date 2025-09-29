#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DATA_SIZE 100
#define MAX_VALUE 100
#define BIN_COUNT 10

int main(int argc, char* argv[]) {
    int rank, size;
    int *data = NULL;
    int local_size;
    int *local_data;
    int local_hist[BIN_COUNT];
    int global_hist[BIN_COUNT];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_size = DATA_SIZE / size;
    local_data = (int*)malloc(local_size * sizeof(int));

    for (int i = 0; i < BIN_COUNT; i++) {
        local_hist[i] = 0;
        global_hist[i] = 0;
    }

    if (rank == 0) {
        data = (int*)malloc(DATA_SIZE * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < DATA_SIZE; i++) {
            data[i] = rand() % (MAX_VALUE + 1);
        }
        printf("Datos generados:\n");
        for (int i = 0; i < DATA_SIZE; i++) {
            printf("%d ", data[i]);
        }
        printf("\n\n");
    }

    MPI_Scatter(data, local_size, MPI_INT,
                local_data, local_size, MPI_INT,
                0, MPI_COMM_WORLD);

    int bin_width = (MAX_VALUE + 1) / BIN_COUNT;
    for (int i = 0; i < local_size; i++) {
        int bin = local_data[i] / bin_width;
        if (bin >= BIN_COUNT) bin = BIN_COUNT - 1;
        local_hist[bin]++;
    }

    MPI_Reduce(local_hist, global_hist, BIN_COUNT, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Histograma final:\n");
        for (int i = 0; i < BIN_COUNT; i++) {
            printf("Bin %d (%d-%d): %d\n",
                   i, i*bin_width, (i+1)*bin_width - 1, global_hist[i]);
        }
        free(data);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}

