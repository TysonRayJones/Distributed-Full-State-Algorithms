#ifndef COMMUNICATION_HPP
#define COMMUNICATION_HPP


#include "types.hpp"
#include "bit_maths.hpp"

#include <mpi.h>



/*
 * MPI environment management 
 */

static void comm_init() {
    int isInit;
    MPI_Initialized(&isInit);
    if (!isInit)
        MPI_Init(NULL, NULL);
}


static void comm_end() {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}


static Nat comm_getRank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return (Nat) rank;
}


static Nat comm_getNumNodes() {
    int numNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    return (Nat) numNodes;
}


static void comm_synch() {
    
    MPI_Barrier(MPI_COMM_WORLD);
}


/*
 * synchronous amplitude exchange
 */

static void comm_exchangeInChunks(Amp* send, Amp* recv, Index numAmps, Nat pairRank) {
    
    int tag = 100;
    Index maxMessageSize = powerOf2(30);
    Index numFullMessages = numAmps / maxMessageSize;
    
    for (Index m=0; m<numFullMessages; m++)
        MPI_Sendrecv(
            &send[m*maxMessageSize], numAmps, MPI_AMP, 
                pairRank, tag,
            &recv[m*maxMessageSize], numAmps, MPI_AMP,
                pairRank, tag,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
    Index remainingIndex = numFullMessages * maxMessageSize;
    Index remainingNumAmps = numAmps % maxMessageSize;
    
    if (remainingNumAmps > 0)
        MPI_Sendrecv(
            &send[remainingIndex], remainingNumAmps, MPI_AMP,
                pairRank, tag, 
            &recv[remainingIndex], remainingNumAmps, MPI_AMP,
                pairRank, tag,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


static void comm_exchangeArrays(
    AmpArray& toSend, Index toSendStartInd, 
    AmpArray& toReceive, Index toReceiveStartInd, 
    Index numAmpsToExchange, Nat pairRank
) {
    comm_exchangeInChunks(
        &toSend[toSendStartInd], &toReceive[toReceiveStartInd],
        numAmpsToExchange, pairRank);
}


static void comm_exchangeArrays(AmpArray& toSend, AmpArray& toReceive, Nat pairRank) {
    
    comm_exchangeArrays(toSend, 0, toReceive, 0, toSend.size(), pairRank);
}



/*
 * asynchronous separate amplitude send and receive
 */
 
static void comm_asynchSendInChunks(Amp* send, Index numAmps, Nat pairRank) {
    
    MPI_Request req;
    int tag = 100;
    
    Index maxMessageSize = powerOf2(30);
    Index numFullMessages = numAmps / maxMessageSize;
    
    for (Index m=0; m<numFullMessages; m++) {
        MPI_Isend(
            &send[m*maxMessageSize], numAmps, MPI_AMP, 
            pairRank, tag, 
            MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);
    }
            
    Index remainingIndex = numFullMessages * maxMessageSize;
    Index remainingNumAmps = numAmps % maxMessageSize;
    
    if (remainingNumAmps > 0) {
        MPI_Isend(
            &send[remainingIndex], remainingNumAmps, MPI_AMP,
            pairRank, tag, 
            MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);
    }
}


static void comm_asynchSendArray(AmpArray& toSend, Index numAmpsToSend, Nat pairRank) {
    
    comm_asynchSendInChunks(toSend.data(), numAmpsToSend, pairRank);
}


static void comm_receiveInChunks(Amp* dest, Index numAmps, Nat pairRank) {
    
    int tag = 100;
    Index maxMessageSize = powerOf2(30);
    Index numFullMessages = numAmps / maxMessageSize;
    
    for (Index m=0; m<numFullMessages; m++)
        MPI_Recv(
            &dest[m*maxMessageSize], numAmps, MPI_AMP, 
            pairRank, tag, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    Index remainingIndex = numFullMessages * maxMessageSize;
    Index remainingNumAmps = numAmps % maxMessageSize;
    
    if (remainingNumAmps > 0)
        MPI_Recv(
            &dest[remainingIndex], remainingNumAmps, MPI_AMP,
            pairRank, tag, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


static void comm_receiveArray(AmpArray& toReceive, Index numAmpsToReceive, Nat pairRank) {
    
    comm_receiveInChunks(toReceive.data(), numAmpsToReceive, pairRank);
}


static void comm_reduceAmp(Amp& localAmp) {

    Amp globalAmp;
    MPI_Allreduce(&localAmp, &globalAmp, 1, MPI_AMP, MPI_SUM, MPI_COMM_WORLD);
    localAmp = globalAmp;
}



#endif // COMMUNICATION_HPP