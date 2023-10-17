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
 * internal convenience functions
 */

static int NULL_TAG = 0;

static void getMessageConfig(Index *messageSize, Index *numMessages, Index numAmps) {

    assert( isPowerOf2(numAmps) );

    // determine the number of max-size messages
    *messageSize = powerOf2(30);
    *numMessages = numAmps / *messageSize; // divides evenly

    // when numAmps < messageSize, we need send only one smaller message (obviously)
    if (*numMessages == 0) {
        *messageSize = numAmps;
        *numMessages = 1;
    }
}



/*
 * synchronous amplitude exchange
 */

static void comm_exchangeInChunks(Amp* send, Amp* recv, Index numAmps, Nat pairRank) {

    // each message is asynchronously dispatched with a final wait, as per arxiv.org/abs/2308.07402

    // divide the data into multiple messages
    Index messageSize, numMessages;
    getMessageConfig(&messageSize, &numMessages, numAmps);

    // each asynch message below will create two requests for subsequent synch
    std::vector<MPI_Request> requests(2*numMessages);

    // asynchronously exchange the messages (effecting MPI_Isendrecv), exploiting orderedness gaurantee
    for (Index m=0; m<numMessages; m++) {
        MPI_Isend(&send[m*messageSize], messageSize, MPI_AMP, pairRank, NULL_TAG, MPI_COMM_WORLD, &requests[2*m]);
        MPI_Irecv(&recv[m*messageSize], messageSize, MPI_AMP, pairRank, NULL_TAG, MPI_COMM_WORLD, &requests[2*m+1]);
    }

    // wait for all exchanges to complete (MPI willl automatically free the request memory)
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
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
 * asynch send and synch receive
 */


static void comm_asynchSendInChunks(Amp* send, Index numAmps, Nat pairRank) {

    // we will not track nor wait for the asynch send; instead, the caller will later comm_synch()
    MPI_Request nullReq;

    // divide the data into multiple messages
    Index messageSize, numMessages;
    getMessageConfig(&messageSize, &numMessages, numAmps);

    // asynchronously send the messages; pairRank receives the same ordering
    for (Index m=0; m<numMessages; m++)
        MPI_Isend(&send[m*messageSize], messageSize, MPI_AMP, pairRank, NULL_TAG, MPI_COMM_WORLD, &nullReq);
}


static void comm_asynchSendArray(AmpArray& toSend, Index numAmpsToSend, Nat pairRank) {
    
    comm_asynchSendInChunks(toSend.data(), numAmpsToSend, pairRank);
}


static void comm_receiveInChunks(Amp* dest, Index numAmps, Nat pairRank) {

    // expect the data in multiple messages
    Index messageSize, numMessages;
    getMessageConfig(&messageSize, &numMessages, numAmps);

    // create a request for each asynch receive below
    std::vector<MPI_Request> requests(numMessages);

    // listen to receive each message asynchronously (as per arxiv.org/abs/2308.07402)
    for (Index m=0; m<numMessages; m++)
        MPI_Irecv(&dest[m*messageSize], messageSize, MPI_AMP, pairRank, NULL_TAG, MPI_COMM_WORLD, &requests[m]);

    // receivers wait for all messages to be received (while sender asynch proceeds)
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}


static void comm_receiveArray(AmpArray& toReceive, Index numAmpsToReceive, Nat pairRank) {
    
    comm_receiveInChunks(toReceive.data(), numAmpsToReceive, pairRank);
}



/*
 * synchronous reduction
 */

static void comm_reduceAmp(Amp& localAmp) {

    Amp globalAmp;
    MPI_Allreduce(&localAmp, &globalAmp, 1, MPI_AMP, MPI_SUM, MPI_COMM_WORLD);
    localAmp = globalAmp;
}



#endif // COMMUNICATION_HPP