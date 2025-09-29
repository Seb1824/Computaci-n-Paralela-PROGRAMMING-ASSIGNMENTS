#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size, n, local_n;
    int *block_data, *cyclic_data;
    double start, end;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(rank==0){
        printf("Ingrese n (divisible entre %d): ", size);
        fflush(stdout);
        scanf("%d",&n);
    }
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    local_n = n/size;

    block_data = (int*)malloc(local_n*sizeof(int));
    srand(time(NULL)+rank);
    for(int i=0;i<local_n;i++) block_data[i] = rank*local_n+i;

    cyclic_data = (int*)malloc(local_n*sizeof(int));

    int* sendcounts = (int*)calloc(size,sizeof(int));
    int* sdispls = (int*)calloc(size,sizeof(int));
    int* recvcounts = (int*)calloc(size,sizeof(int));
    int* rdispls = (int*)calloc(size,sizeof(int));

    start = MPI_Wtime();
    for(int p=0;p<size;p++){
        sendcounts[p]=0;
        recvcounts[p]=0;
    }
    for(int i=0;i<local_n;i++){
        int global_index = rank*local_n+i;
        int dest = global_index % size;
        sendcounts[dest]++;
    }
    sdispls[0]=0; rdispls[0]=0;
    for(int p=1;p<size;p++){
        sdispls[p]=sdispls[p-1]+sendcounts[p-1];
        rdispls[p]=rdispls[p-1]+recvcounts[p-1];
    }
    int* sendbuf = (int*)malloc(local_n*sizeof(int));
    int* scount = (int*)calloc(size,sizeof(int));
    for(int i=0;i<local_n;i++){
        int global_index = rank*local_n+i;
        int dest = global_index % size;
        int pos = sdispls[dest]+scount[dest];
        sendbuf[pos] = block_data[i];
        scount[dest]++;
    }
    MPI_Alltoall(sendcounts,1,MPI_INT,recvcounts,1,MPI_INT,MPI_COMM_WORLD);
    rdispls[0]=0;
    for(int p=1;p<size;p++) rdispls[p]=rdispls[p-1]+recvcounts[p-1];
    MPI_Alltoallv(sendbuf,sendcounts,sdispls,MPI_INT,
                  cyclic_data,recvcounts,rdispls,MPI_INT,MPI_COMM_WORLD);
    end = MPI_Wtime();
    if(rank==0) printf("Tiempo Block->Cyclic: %f segundos\n", end-start);

    start = MPI_Wtime();
    for(int p=0;p<size;p++){
        sendcounts[p]=0;
        recvcounts[p]=0;
    }
    for(int i=0;i<local_n;i++){
        int global_index = i*size+rank;
        int dest = global_index/local_n;
        sendcounts[dest]++;
    }
    sdispls[0]=0; rdispls[0]=0;
    for(int p=1;p<size;p++){
        sdispls[p]=sdispls[p-1]+sendcounts[p-1];
        rdispls[p]=rdispls[p-1]+recvcounts[p-1];
    }
    scount = (int*)calloc(size,sizeof(int));
    for(int i=0;i<local_n;i++){
        int global_index = i*size+rank;
        int dest = global_index/local_n;
        int pos = sdispls[dest]+scount[dest];
        sendbuf[pos] = cyclic_data[i];
        scount[dest]++;
    }
    MPI_Alltoall(sendcounts,1,MPI_INT,recvcounts,1,MPI_INT,MPI_COMM_WORLD);
    rdispls[0]=0;
    for(int p=1;p<size;p++) rdispls[p]=rdispls[p-1]+recvcounts[p-1];
    MPI_Alltoallv(sendbuf,sendcounts,sdispls,MPI_INT,
                  block_data,recvcounts,rdispls,MPI_INT,MPI_COMM_WORLD);
    end = MPI_Wtime();
    if(rank==0) printf("Tiempo Cyclic->Block: %f segundos\n", end-start);

    free(block_data);
    free(cyclic_data);
    free(sendcounts);
    free(sdispls);
    free(recvcounts);
    free(rdispls);
    free(sendbuf);
    free(scount);

    MPI_Finalize();
    return 0;
}
