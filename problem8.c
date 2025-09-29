#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int cmpfunc(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int* merge(int* arr1, int n1, int* arr2, int n2) {
    int* result = (int*)malloc((n1+n2)*sizeof(int));
    int i=0,j=0,k=0;
    while (i<n1 && j<n2) {
        if (arr1[i] <= arr2[j]) result[k++] = arr1[i++];
        else result[k++] = arr2[j++];
    }
    while (i<n1) result[k++] = arr1[i++];
    while (j<n2) result[k++] = arr2[j++];
    return result;
}

int main(int argc, char* argv[]) {
    int rank, size, n, local_n;
    int *local_arr, *gathered;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (rank==0) {
        printf("Ingrese n (divisible entre %d): ", size);
        fflush(stdout);
        scanf("%d",&n);
    }
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

    local_n = n/size;
    local_arr = (int*)malloc(local_n*sizeof(int));

    srand(time(NULL)+rank);
    for(int i=0;i<local_n;i++) local_arr[i] = rand()%100;

    qsort(local_arr, local_n, sizeof(int), cmpfunc);

    if (rank==0) gathered = (int*)malloc(n*sizeof(int));
    MPI_Gather(local_arr, local_n, MPI_INT, gathered, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank==0) {
        printf("Listas locales ordenadas:\n");
        for(int i=0;i<n;i++) printf("%d ", gathered[i]);
        printf("\n");
        free(gathered);
    }

    int step=1;
    int* merged = local_arr;
    int merged_size = local_n;
    while(step<size){
        if(rank%(2*step)==0){
            if(rank+step<size){
                int recv_size;
                MPI_Recv(&recv_size,1,MPI_INT,rank+step,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                int* recv_data=(int*)malloc(recv_size*sizeof(int));
                MPI_Recv(recv_data,recv_size,MPI_INT,rank+step,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                int* new_merge=merge(merged,merged_size,recv_data,recv_size);
                free(merged);
                free(recv_data);
                merged=new_merge;
                merged_size+=recv_size;
            }
        } else {
            int dest=rank-step;
            MPI_Send(&merged_size,1,MPI_INT,dest,0,MPI_COMM_WORLD);
            MPI_Send(merged,merged_size,MPI_INT,dest,0,MPI_COMM_WORLD);
            free(merged);
            break;
        }
        step*=2;
    }

    if(rank==0){
        printf("Lista global ordenada:\n");
        for(int i=0;i<merged_size;i++) printf("%d ", merged[i]);
        printf("\n");
        free(merged);
    }

    MPI_Finalize();
    return 0;
}
