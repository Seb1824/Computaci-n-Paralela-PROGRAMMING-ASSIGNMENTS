#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size;
    long long int tosses, local_tosses;
    long long int local_in_circle = 0;
    long long int global_in_circle = 0;
    double x, y, distance_squared;
    double pi_estimate;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Ingrese el número total de lanzamientos: ");
        fflush(stdout);
        scanf("%lld", &tosses);
    }

    MPI_Bcast(&tosses, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    local_tosses = tosses / size;
    srand(time(NULL) + rank); 

    for (long long int toss = 0; toss < local_tosses; toss++) {
        x = (2.0 * rand() / RAND_MAX) - 1.0; 
        y = (2.0 * rand() / RAND_MAX) - 1.0; 
        distance_squared = x*x + y*y;
        if (distance_squared <= 1.0) {
            local_in_circle++;
        }
    }
 
    MPI_Reduce(&local_in_circle, &global_in_circle, 1, MPI_LONG_LONG_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi_estimate = 4.0 * ((double) global_in_circle) / ((double) tosses);
        printf("Estimación de PI con %lld lanzamientos: %f\n", tosses, pi_estimate);
    }

    MPI_Finalize();
    return 0;
}
