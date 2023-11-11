//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Weile Jia

/// @file device_mpi_interf.cpp
/// @brief Inerface with MPI to facilitate communication.
/// @date 2020-08-23
#include  "device_mpi_interf.h"
#include  "utility.h"
static int stringCmp(const void *a, const void* b)
{
	char* m = (char*)a;
	char* n = (char*)b;
	int i, sum = 0;
	
	for( i = 0; i < MPI_MAX_PROCESSOR_NAME; i++ )
		if (m[i] == n[i])
			continue;
		else
		{
			sum = m[i] - n[i];
			break;
		}		
	
	return sum;
}
namespace scales{

// *********************************************************************
// Constants
// *********************************************************************

namespace device_mpi{


void setDevice(MPI_Comm comm)
{
    Int nprocs, rank, namelen, n, color, myrank, dev;
    char host_name[MPI_MAX_PROCESSOR_NAME];
    char (*host_names)[MPI_MAX_PROCESSOR_NAME];
    size_t bytes;
    struct cudaDeviceProp deviceProp;
    MPI_Comm nodeComm;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Get_processor_name(host_name,&namelen);
    
     bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
     host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
     strcpy(host_names[rank], host_name);
     
     /* get all the hostnames on all the nodes */
     for (n=0; n<nprocs; n++)
       MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, comm); 

     qsort(host_names, nprocs,  sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);
     color = 0;

     for (n=0; n<nprocs-1; n++)
     {
       if(strcmp(host_name, host_names[n]) == 0)
       {
          break;
       }
       if(strcmp(host_names[n], host_names[n+1])) 
       {
          color++;
       }
     }
     int mysize;
     MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
     MPI_Comm_rank(nodeComm, &myrank);
     MPI_Comm_size(nodeComm, &mysize);

     int deviceCount,slot = 0;
     int *devloc;
     cudaGetDeviceCount(&deviceCount);

     devloc=(int *)malloc(mysize*sizeof(int));
     for (dev = 0; dev < mysize; ++dev)
       devloc[dev] = 0;
     
     for (dev = 0; dev < deviceCount; ++dev)
       devloc[slot++]=dev;
     
     if( (devloc[myrank] >= deviceCount) )
     {
       printf ("Error:::Assigning device %d  to process on node %s rank %d \n",devloc[myrank],  host_name, rank );
       MPI_Abort(MPI_COMM_WORLD, MPI_ERR_TYPE);
       MPI_Abort(MPI_COMM_WORLD, MPI_ERR_TYPE);
     }
     
     printf ("Assigning device %d  to process on node %s rank %d, OK\n",devloc[myrank],  host_name, rank );
     cudaSetDevice(devloc[myrank]);
     fflush(stdout);
     free(devloc);
     free(host_names);
}

} // namespace mpi

} // namespace scales
