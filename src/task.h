/*
 * task.h
 *
 *  Created on: Mar 3, 2014
 *      Author: zdravko
 */

#ifndef TASK_H_
#define TASK_H_

#include <iostream> // because of writing
#include <stdio.h>
#include <math.h> // for modf function
#include <ctime> // for the timestamp() function
#include <vector>
#include <string> // for converting numbers to strings (needed for the HTML file for visualization)
#include <unistd.h> // for sleep()
#include "utility.h"

class Task
{
	private:
		int comp_amount;
		int id;
		void execute_load(int); // called by compute()
	public:
		Task ();
		void setID(int); // set id
		int getID(); // return id
		int getProcessID(); // return rank of the process running this task
		void waitUnlocksViaRecv(int **commMatrix, int **schedule, double stamp);
		void sendUnlocks(int **commMatrix, int **schedule, MPI_Request& request, double);
		void communicate(int **commMatrix, int **schedule, int *comDataChunk, double stamp); // comm phase (MIXED GRAPH ONLY)
		void recvCommDAGSingleBuffers(double **commMatrix, int **schedule, int *commDataChunkRecv, double stamp, std::string &htmlFileString); // comm phase (DAG ONLY)
		std::size_t getTotalNumOfSendChunks(double **commMatrix);
		void recvCommDAGAllBuffers(double **commMatrix, int **schedule, int *commDataChunkRecv, double stamp); // comm phase (DAG ONLY)
		void sendCommDAGSingleBuffers(double **commMatrix, int **schedule, int *commDataChunkSend, MPI_Request *sendRequests, std::size_t &sendRequestsCounter, double stamp); // comm phase (DAG ONLY)
		void sendCommDAGAllBuffers(double **commMatrix, int **schedule, int *commDataChunkSend, double stamp); // comm phase (DAG ONLY)
		void compute(int **compAmounts, double stamp, std::string &htmlFileString); // compute phase
		void comp_bench(int mpi_rank, double*);
		void comm_bench(int mpi_rank, int mpi_size, int* comm_chunk, int comm_chunk_size, double*, double*);
		int findHPUForTaskInSchedule(int **schedule, std::size_t numOfHPUs, std::size_t maxNumTasksInSchedule, int taskID);
		void load_data(int); // load from storage [FUTURE]
		void store_data(int); // store to storage [FUTURE]
};

/****
 * Initialize computation load if no load is given
 * ****/
Task::Task()
{
	comp_amount = 1000;
	id=0;
}

/**** Function which sets object (Task) id
 * (identifies from the allocation vector)
 * ****/
void Task::setID(int id)
{
	this->id=id;
}

/**** Function which returns the object (Task) id
 * ****/
int Task::getID()
{
	return this->id;
}

/**** Returning rank of the process executing task
 * not in use still
 * ****/
int Task::getProcessID()
{
	//return MPI::COMM_WORLD.Get_rank();
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;
}

/**** Receive unlock
 * TODO: Only for MIXED GRAPH for now
 * Receive all precedence constraints by a task calling this function
 * Checks column of an alloc matrix, if a value is 0 this means that task depends
 * on the task corresponding to the row number,
 * then receive prec from the process assigned to the precedence task
 *
 * This method should be invoked per task!
 * ****/
void Task::waitUnlocksViaRecv(int **commMatrix, int **schedule, double stamp)
{
	int dummy = 0; //variable received as a precedence semaphore
	int taskID = this->getID();
	for (std::size_t i = 0; i < numOfTasks; i++)
	{
		if (commMatrix[i][taskID] == 0)
		{
			printTimestamp(stamp);
			int hpuOfPrecedentTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);
/*DEBUG*/std::cout << "\tP" << this->getProcessID() << ":\tT" << taskID << ": waits for unlock from T" << i << " (P" << hpuOfPrecedentTask << ")" << std::endl;
			if ( this->getProcessID() != hpuOfPrecedentTask ) // receive unlock only if sent from another process (not to receive from itself)
				MPI_Recv(&dummy, 1, MPI_INT, hpuOfPrecedentTask, taskID, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printTimestamp(stamp);
/*DEBUG*/std::cout << "\tP" << this->getProcessID() << ":\tT" << taskID << ": got unlocked by T" << i << " (P" << hpuOfPrecedentTask << ")" << std::endl;
		}
	}
}

/**** Send unlock
 * TODO: Only for MIXED GRAPH for now
 * Send every successor a signal that his precedence is unlocked by a task calling this function
 * Checks column of an alloc matrix, if a value is 0 this means that task depends
 * on the task corresponding to the row number,
 * then receive prec from the PROCESS assigned to the precedence task
 * ****/
void Task::sendUnlocks(int **commMatrix, int **schedule, MPI_Request& request, double stamp)
{
	int dummy = 0; //variable sent as a precedence semaphore
	int taskID = this->getID();
	for (int i = 0; i < numOfTasks; i++)
	{
		if (commMatrix[taskID][i] == 0)
		{
			printTimestamp(stamp);
			int hpuOfChildTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);
/*DEBUG*/	std::cout << "\tP" << ":\tT" << taskID << ": unlocking T" << i << " (P" << hpuOfChildTask << ")" << std::endl;
			if (this->getProcessID() != hpuOfChildTask) // send unlock only to another process (not to send to itself)
				MPI_Isend(&dummy, 1, MPI_INT, hpuOfChildTask, i, MPI_COMM_WORLD, &request);
			printTimestamp(stamp);
/*DEBUG*/std::cout << "\tP" << this->getProcessID() << ":\tT" << taskID << ": unlocked T" << i << " (P" << hpuOfChildTask << ")" << std::endl;
		}
	}
}

/****Communication phase of a task
 * TODO: Only for MIXED GRAPH for now
 * ASSUMPTION: the task computes first, than communicates.
 * 				Having communication started first (non-blocking) and then computing would be much better
 * 				But that depends on what we want to show.
 * SEND if lower number than one inspected
 * RECEIVE if higher number than one inspected
 * Since the comm amount is double:
 * 		The whole part determines how many times a 10MB data chunk is sent (the benchmark is obtained for a 10MB tranfer)
 * 		The decimal part is converted so that only a fraction of 10MB is sent
 * TODO: Since this comm is for MIXED graph, the comm should be synchronized (the communication requires
 * 			active participation of all tasks that communicate - a partition)
 * 			- this should involve using collective comm
 * TODO: VARIATIONS:
 * 			1. Both sends and receives are non blocking
 * 			2. Both sends and receives are blocking
 * 			3. Sends are blocking, receives are non blocking
 * 			4. Sends are non blocking, receives are blocking
 * 			SUB-VARIATIONS:
 *
 * TODO: NOT FINISHED! A lot to think about here:
 * 						Start all comm (non-blocking) or
 * 						Start all com (blocking) or
 * 						Start certain comm (blocking or not)
 * 						etc...
 * ****/
void Task::communicate(int **commMatrix, int **schedule, int *commDataChunk, double stamp)
{
//	int taskID = this->getID();
//
//	std::vector<MPI_Request*> commSendRequestsGlobal;
//	std::vector<MPI_Request*> commRecvRequestsGlobal;
//
//	std::vector<std::size_t> numberOfSendChunks;
//	std::vector<std::size_t> numberOfRecvChunks;
//
//	std::size_t sendCounter = 0;
//	std::size_t recvCounter = 0;
//
//	for (int i=0; i<numOfTasks; i++)
//	{
//		if (commMatrix[taskID][i] > 0) // > 0 is comm
//		{
//			/*** TODO: ***WRONG***, there is no need for checking this! ***ERROR*** THE SEND PART ***/
//			if (taskID < i)
//			{
//				int decimalPart = 0;
//				printTimestamp(stamp);
//	/*DEBUG*/	std::cout << "\tT" << taskID << " start send to T" << i << std::endl;
//				int hpuOfReceiverTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);
//
//				// split the amount of comm to a whole and decimal part
//				int commWholePart = (long)commMatrix[taskID][i]; // the type cast is used to get the whole number (to separate it from the decimal part)
//				float commDecimalPart = commMatrix[taskID][i] - (long)commMatrix[taskID][i]; // the decimal part is the difference between the number and its whole part
//
//				// if decimal part exists (it is unlikely that it doesn't (for example that amount of comm is 18.00000), but nevertheless)
//				if (commDecimalPart >= 1)
//					decimalPart = 1;
//
//				// create an array of requests for sends
//				MPI_Request *commSendRequests = new MPI_Request[commWholePart + decimalPart]; // N sends for whole parts + 1 send for decimal part (if it exists)
//
//				// send the amount of data necessary (the whole)
//				for (std::size_t j = 0; j < commWholePart; j++)
//					MPI_Isend(commDataChunk, COMM_CHUNK_SIZE, MPI_INT, hpuOfReceiverTask, this->getProcessID(), MPI_COMM_WORLD, &commSendRequests[j]);
//
//				// send the decimal part (if there is one)
//				if (commDecimalPart >= 1)
//					MPI_Isend(commDataChunk, (long)((commDecimalPart)*COMM_CHUNK_SIZE), MPI_INT, hpuOfReceiverTask, this->getProcessID(), MPI_COMM_WORLD, &commSendRequests[commWholePart]);
//
//				// push a vector of requests for all sends to this destination into a vector
//				commSendRequestsGlobal.push_back(commSendRequests);
//
//				// push a number of chunks to a vector (number of chunks for this destination)
//				numberOfSendChunks.push_back(commWholePart + decimalPart);
//
//				// count the number of destinations for sending
//				sendCounter++;
//
//			}
//			/*** THE RECEIVE PART - Analogous to the send part, so no comments) ***/
//			else if (taskID > i) // we don't take into account the elements on a main diagonal, or?
//			{
//				int decimalPart = 0;
//				printTimestamp(stamp);
//	/*DEBUG*/	std::cout << "\tT" << taskID << " start receive from T" << i << std::endl;
//				int hpuOfSenderTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);
//
//				int commWholePart = (long)commMatrix[taskID][i];
//				float commDecimalPart = commMatrix[taskID][i] - (long)commMatrix[taskID][i];
//
//				if (commDecimalPart >= 1)
//					decimalPart = 1;
//
//				MPI_Request *commRecvRequests = new MPI_Request[commWholePart + decimalPart]; // N receives for whole parts + 1 receive for decimal part (if it exists)
//
//				for (std::size_t j = 0; j < commWholePart; j++)
//					MPI_Irecv(commDataChunk, COMM_CHUNK_SIZE, MPI_INT, hpuOfSenderTask, hpuOfSenderTask, MPI_COMM_WORLD, &commRecvRequests[j]);
//
//				if (commDecimalPart >= 1)
//					MPI_Irecv(commDataChunk, COMM_CHUNK_SIZE, MPI_INT, hpuOfSenderTask, hpuOfSenderTask, MPI_COMM_WORLD, &commRecvRequests[commWholePart]);
//
//				commRecvRequestsGlobal.push_back(commRecvRequests);
//
//				// push a number of chunks to a vector (number of chunks for this source)
//				numberOfRecvChunks.push_back(commWholePart + decimalPart);
//
//				// count the number of destinations for sending
//				recvCounter++;
//
//				printTimestamp(stamp);
//	/*DEBUG*/	std::cout << "\tT" << taskID << " finish receive from T" << i << std::endl;
//			}
//		}
//	}
//
//	// TODO: waiting in batches (separately for sends and receives), or post all waits?
//
//	// wait for sends
//	for (int i = 0; i < sendCounter; i++)
//	{
//		MPI_Waitall(numberOfSendChunks[i], commSendRequestsGlobal[i], MPI_STATUSES_IGNORE);
//		printTimestamp(stamp);
///*DEBUG*/	std::cout << "\tT" << taskID << " finish send to T" << i << std::endl;
//	}
//
//	// wait for receives
//	for (int i = 0; i < recvCounter; i++)
//		MPI_Waitall(numberOfRecvChunks[i], commRecvRequestsGlobal[i], MPI_STATUSES_IGNORE);
}

/**** Counting the total required number of sends for each tasks (multiple destination tasks included)
 * For now used only to calculate the necessary number of MPI_Request array elements
 * */
std::size_t Task::getTotalNumOfSendChunks(double **commMatrix)
{
	int taskID = this->getID();

	std::size_t totalChunksToSend = 0;

    for (int i = 0; i < numOfTasks; i++)
    {
        if (commMatrix[taskID][i] >= 0) // >= 0 is comm
        {
            if (taskID != i) // we don't take into account the elements on a main diagonal, or?
            {
                // Separating decimal and whole part of the communication amount
                // 1. Temp variables to store int and decimal parts (all have to be double)
                double wholePartOfComm, decimalPartOfComm, temp;
                // 2. Splitting the comm
                decimalPartOfComm = modf (commMatrix[taskID][i], &wholePartOfComm);
                // 3. Storing the int part
                totalChunksToSend += wholePartOfComm;
                // 4. Multiplying the decimal part with the chunk size (to get that part of the 10MB chunk), and saving only the int part of the result (the MPI_Send cannot have a decimal count parameter), temp is not used later on
                temp = modf ((decimalPartOfComm * COMM_CHUNK_SIZE), &wholePartOfComm);
                // 5. Storing the partial comm chunk size (number of ints to send)

                if ( wholePartOfComm > 0)
                {
                    totalChunksToSend++;
                }
            }
        }
    }
    return totalChunksToSend;
}

/**** Receive communication from all the predecessors, this also unlocks task
 * Uses buffers for ALL receives
 * For DAG: receiving data & unlocks task
 * TODO: Can it be used for the MIXED graphs?
 *
 * TODO: Only for DAG for now
 * Receive all precedence constraints by a task calling this function
 * Checks column of an alloc matrix, if a value is 0 or greater this means
 * that task receives data from the task corresponding to the row number,
 * then receive data from the process assigned to the precedence task
 *
 * TODO: rotate receives from all predecessors (to receive data ASAP)
 * 		 (when waiting for unlocks only the order of receives is not relevant (because receive time is negligible), but when there is comm it IS relevant)
 * This method should be invoked per task!
 *
 * TODO: CHANGEABLES:
 * 			1. There can be a comm amount of 0 in the comm matrix (I don't think so, but the code can cope with that)
 * 			2. The task sends data to another task even if they are on the same HPU, because we are simulating real world behavior
 * ****/
void Task::recvCommDAGAllBuffers(double **commMatrix, int **schedule, int *commDataChunkRecv, double stamp)
{
	int taskID = this->getID();

//	std::ofstream outfile;// declaration of file pointer named outfile
//	std::string filename = "IN_T" + std::to_string(taskID) + ".txt";
//	outfile.open(filename, std::ios::out); // opens file named "filename" for output

	// initially create an array of data about chunks received from each source task, one instance of struct per source task
	struct receivesFromASource
	{
		std::size_t senderTaskID;
		std::size_t hpuOfSenderTask;
		std::size_t numOfWholeChunks;
		std::size_t partialChunkSize;
		bool hasPartialChunk = false;
	};
	std::vector<receivesFromASource> receives;
	std::vector<receivesFromASource>::iterator receivesIterator;
	std::size_t receivesCounter = 0;
	std::size_t totalChunksToReceive = 0;

	std::cout << "T" << taskID << "(P" << this->getProcessID() << ")\n";

	std::size_t bufferArrayNoOfElements = 0; /*DEBUG*/

	for (int i = 0; i < numOfTasks; i++)
	{
		if (commMatrix[i][taskID] >= 0) // >= 0 is comm
		{
			if (taskID != i) // we don't take into account the elements on a main diagonal, or?
			{
				receives.push_back(receivesFromASource()); // add a blank chunk

				// populate the blank chunk with data
				receives[receivesCounter].senderTaskID = i;
				// Even if both tasks reside on the same HPU, the data is sent - behavior of the real apps
				receives[receivesCounter].hpuOfSenderTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);

				std::cout << "T" << i << "(P" << receives[receivesCounter].hpuOfSenderTask << ")\n";

				// Separating decimal and whole part of the communication amount
				// 1. Temp variables to store int and decimal parts (all have to be double)
				double wholePartOfComm, decimalPartOfComm, temp;
				// 2. Splitting the comm
				decimalPartOfComm = modf (commMatrix[i][taskID], &wholePartOfComm);
				// 3. Storing the int part
				receives[receivesCounter].numOfWholeChunks = wholePartOfComm;
				// 4. Multiplying the decimal part with the chunk size (to get that part of the 10MB chunk), and saving only the int part of the result (the MPI_Recv cannot have a decimal count parameter), temp is not used later on
				temp = modf ((decimalPartOfComm * COMM_CHUNK_SIZE), &wholePartOfComm);
				// 5. Storing the partial comm chunk size (number of ints to recv)
				receives[receivesCounter].partialChunkSize = wholePartOfComm;

				totalChunksToReceive += receives[receivesCounter].numOfWholeChunks;

				bufferArrayNoOfElements += (receives[receivesCounter].numOfWholeChunks * COMM_CHUNK_SIZE) + receives[receivesCounter].partialChunkSize; /*DEBUG*/

				if (receives[receivesCounter].partialChunkSize > 0)
				{
					receives[receivesCounter].hasPartialChunk = true;
					totalChunksToReceive++;
				}

				std::cout << "RECV: T" << taskID << "(P" << this->getProcessID() << ") from T" << i << "(P" << receives[receivesCounter].hpuOfSenderTask << "): chunks: " << receives[receivesCounter].numOfWholeChunks << " + " << ((wholePartOfComm > 0) ? 1 : 0) << "\n";
				receivesCounter++;
			}
		}
	}

	std::cout << "Task " << taskID << " recv buffer: " << bufferArrayNoOfElements << ", size in MB: " << ((bufferArrayNoOfElements * 4)/1024)/1024 << std::endl;

	// create receive buffer (has to be here, since now we know the necessary size)
	int **recvBuffer;
	// new row per each sender
	recvBuffer = new int*[receivesCounter];
	// calculate sizes of the recv buffer rows (for each sender)
	for (std::size_t i = 0; i < receivesCounter; i++)
	{
		std::size_t rowSize = (receives[receivesCounter].numOfWholeChunks * COMM_CHUNK_SIZE) + receives[receivesCounter].partialChunkSize;
		recvBuffer[i] = new int[rowSize];
	}
	MPI_Request *commRecvRequests = new MPI_Request[receivesCounter]; // N receives for whole parts + 1 receive for decimal part (if it exists)

	receivesCounter = 0; // reset for reuse

	// process each source that sends data to this task
	for (receivesIterator = receives.begin(); receivesIterator != receives.end(); ++receivesIterator)
	{
		printTimestamp(stamp);
/*DEBUG*/	std::cout << "\tT" << taskID << " start receive from T" << receivesIterator->senderTaskID << std::endl;

		std::size_t chunkSize = (receivesIterator->numOfWholeChunks * COMM_CHUNK_SIZE) + receivesIterator->partialChunkSize;
		MPI_Irecv(recvBuffer[receivesCounter], chunkSize, MPI_INT, receivesIterator->hpuOfSenderTask, MPI_ANY_TAG, MPI_COMM_WORLD, &commRecvRequests[receivesCounter]);

		receivesCounter++;

	} // all receives posted

	// WAITING
	// wait for receives
	/*DEBUG*/ std::cout << "\tT" << taskID << ": receives posted.\n";

	MPI_Waitall(receivesCounter, commRecvRequests, MPI_STATUSES_IGNORE);


	// release memory
	delete[] commRecvRequests;
	/***NEW RECEIVE MECHANISM***/
	for (std::size_t i = 0; i < receivesCounter; i++)
		delete[] recvBuffer[i];
	delete[] recvBuffer;
	/***NEW RECEIVE MECHANISM***/

//	outfile.close();// closes file

	printTimestamp(stamp);
///*DEBUG*/	std::cout << "\tT" << taskID << " finish receive from T" << i << std::endl;
	std::cout << "\tT" << taskID << " finished all receives." << std::endl;

}

/**** Send unlock
 * Uses SEPARATE buffers for ALL sends
 * TODO: Only for MIXED GRAPH for now
 * Send every successor a signal that his precedence is unlocked by a task calling this function
 * Checks column of an alloc matrix, if a value is 0 this means that task depends
 * on the task corresponding to the row number,
 * then receive prec from the PROCESS assigned to the precedence task
 * ****/
void Task::sendCommDAGAllBuffers(double **commMatrix, int **schedule, int *commDataChunkSend, double stamp)
{

    int taskID = this->getID();

    // initially create an array of data about chunks sent to each destination task, one instance of struct per source task
    struct sendsToADestination
    {
        std::size_t receiverTaskID;
        std::size_t hpuOfReceiveTask;
        std::size_t numOfWholeChunks;
        std::size_t partialChunkSize = 0;
        bool hasPartialChunk = false;
    };
    std::vector<sendsToADestination> sends;
    std::vector<sendsToADestination>::iterator sendsIterator;
    std::size_t sendsCounter = 0;
    std::size_t totalChunksToSend = 0;

    std::cout << "T" << taskID << "(P" << this->getProcessID() << ")\n";

    std::size_t bufferArrayNoOfElements = 0; /*DEBUG*/

    for (int i = 0; i < numOfTasks; i++)
    {
        if (commMatrix[taskID][i] >= 0) // >= 0 is comm
        {
            if (taskID != i) // we don't take into account the elements on a main diagonal, or?
            {
                sends.push_back(sendsToADestination()); // add a blank chunk

                // populate the blank chunk with data
                sends[sendsCounter].receiverTaskID = i;
                // Even if both tasks reside on the same HPU, the data is sent - behavior of the real apps
                sends[sendsCounter].hpuOfReceiveTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);

                std::cout << "T" << i << "(P" << sends[sendsCounter].hpuOfReceiveTask << ")\n";

                // Separating decimal and whole part of the communication amount
                // 1. Temp variables to store int and decimal parts (all have to be double)
                double wholePartOfComm, decimalPartOfComm, temp;
                // 2. Splitting the comm
                decimalPartOfComm = modf (commMatrix[taskID][i], &wholePartOfComm);
                // 3. Storing the int part
                sends[sendsCounter].numOfWholeChunks = wholePartOfComm;
                // 4. Multiplying the decimal part with the chunk size (to get that part of the 10MB chunk), and saving only the int part of the result (the MPI_Send cannot have a decimal count parameter), temp is not used later on
                temp = modf ((decimalPartOfComm * COMM_CHUNK_SIZE), &wholePartOfComm);
                // 5. Storing the partial comm chunk size (number of ints to send)
                sends[sendsCounter].partialChunkSize = wholePartOfComm;

                totalChunksToSend += sends[sendsCounter].numOfWholeChunks;

                bufferArrayNoOfElements += (sends[sendsCounter].numOfWholeChunks * COMM_CHUNK_SIZE) + sends[sendsCounter].partialChunkSize; /*DEBUG*/

                if ( sends[sendsCounter].partialChunkSize > 0)
                {
                    sends[sendsCounter].hasPartialChunk = true;
                    totalChunksToSend++;
                }

                sendsCounter++;
            }
        }
    }

    std::cout << "Task " << taskID << " send buffer: " << bufferArrayNoOfElements << ", size in MB: " << ((bufferArrayNoOfElements * 4)/1024)/1024 << std::endl;

    // create send buffer (has to be here, since now we know the necessary size)
    int **sendBuffer;
    // new row per each sender
    sendBuffer = new int*[sendsCounter];
    // calculate sizes of the recv buffer rows (for each sender)
    for (std::size_t i = 0; i < sendsCounter; i++)
    {
        std::size_t rowSize = (sends[sendsCounter].numOfWholeChunks * COMM_CHUNK_SIZE) + sends[sendsCounter].partialChunkSize;
        sendBuffer[i] = new int[rowSize];
    }
    MPI_Request *commSendRequests = new MPI_Request[sendsCounter]; // N receives for whole parts + 1 receive for decimal part (if it exists)

    // the total number of chunks received from all sources (necessary to store requests in a global array of receives for this task)
    int chunkCounter = 0;

    sendsCounter = 0;

    // process each source that sends data to this task
    for (sendsIterator = sends.begin(); sendsIterator != sends.end(); ++sendsIterator)
    {
        printTimestamp(stamp);
/*DEBUG*/   std::cout << "\tT" << taskID << " start send to T" << sendsIterator->receiverTaskID << std::endl;

        std::size_t chunkSize = (sendsIterator->numOfWholeChunks * COMM_CHUNK_SIZE) + sendsIterator->partialChunkSize;
        MPI_Isend(sendBuffer[sendsCounter], chunkSize, MPI_INT, sendsIterator->hpuOfReceiveTask, sendsCounter, MPI_COMM_WORLD, &commSendRequests[sendsCounter]);

        sendsCounter++;

    } // all receives posted

    // WAITING
    // wait for sends
    /*DEBUG*/ std::cout << "\tT" << taskID << ": sends posted.\n";
    int flag;

//    MPI_Testall(chunkCounter, commSendRequests, &flag, MPI_STATUSES_IGNORE);
//    while (!flag)
//    {
//      for (int i = 0; i < chunkCounter; i++)
//      {
//          int flag;
//          MPI_Status status;
//          MPI_Test(&(commSendRequests[i]), &flag, &status);
//          outfile << taskID << "\t" << this->getProcessID() << "\t" << status.MPI_SOURCE << "\t" << status.MPI_TAG << "\t" << ((flag) ? "1" : "0") << std::endl;
//      }
//      sleep(1);
//        MPI_Testall(chunkCounter, commSendRequests, &flag, MPI_STATUSES_IGNORE);
//    }

    MPI_Waitall(sendsCounter, commSendRequests, MPI_STATUSES_IGNORE);

    // release memory
    delete[] commSendRequests;
    for (std::size_t i = 0; i < sendsCounter; i++)
        delete[] sendBuffer[i];
    delete[] sendBuffer;

    printTimestamp(stamp);
///*DEBUG*/ std::cout << "\tT" << taskID << " finish receive from T" << i << std::endl;
    std::cout << "\tT" << taskID << " finished all sends." << std::endl;

//  int dummy = 0; //variable sent as a precedence semaphore
//  int taskID = this->getID();
//  for (int i = 0; i < numOfTasks; i++)
//  {
//      if (commMatrix[taskID][i] == 0)
//      {
//          printTimestamp(stamp);
//          int hpuOfChildTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);
///*DEBUG*/ std::cout << "\tP" << ":\tT" << taskID << ": unlocking T" << i << " (P" << hpuOfChildTask << ")" << std::endl;
//          if (this->getProcessID() != hpuOfChildTask) // send unlock only to another process (not to send to itself)
//              MPI_Isend(&dummy, 1, MPI_INT, hpuOfChildTask, i, MPI_COMM_WORLD, &request);
//          printTimestamp(stamp);
///*DEBUG*/std::cout << "\tP" << this->getProcessID() << ":\tT" << taskID << ": unlocked T" << i << " (P" << hpuOfChildTask << ")" << std::endl;
//      }
//  }
}

/**** Receive communication from all the predecessors, this also unlocks task
 * Uses SINGLE 10MB buffer which is reused constantly
 * For DAG: receiving data & unlocks task
 * TODO: Can it be used for the MIXED graphs?
 *
 * TODO: Only for DAG for now
 * Receive all precedence constraints by a task calling this function
 * Checks column of an alloc matrix, if a value is 0 or greater this means
 * that task receives data from the task corresponding to the row number,
 * then receive data from the process assigned to the precedence task
 *
 * Rotate receives from all predecessors (to receive data ASAP) - done by using non-blocking comm and MPI_Waitall
 * 		 (when waiting for unlocks only the order of receives is not relevant (because receive time is negligible),
 * 		 but when there is comm it IS relevant)
 * This method should be invoked per task!
 *
 * TODO: CHANGEABLES:
 * 			1. There can be a comm amount of 0 in the comm matrix (I don't think so, but the code can cope with that)
 * 			2. The task sends data to another task even if they are on the same HPU, because we are simulating real world behavior
 * ****/
void Task::recvCommDAGSingleBuffers(double **commMatrix, int **schedule, int *commDataChunkRecv, double stamp, std::string &htmlFileString)
{
	int taskID = this->getID();

	// Stringstreams for populating the HTML output
	std::stringstream ssProcessID, ssTaskID, ssStart, ssStop;
	ssProcessID << this->getProcessID();
	ssTaskID << taskID;

	// initially create an array of data about chunks received from each source task, one instance of struct per source task
	struct receivesFromASource
	{
		std::size_t senderTaskID;
		std::size_t hpuOfSenderTask;
		std::size_t numOfWholeChunks;
		std::size_t partialChunkSize;
		bool hasPartialChunk = false;
	};
	std::vector<receivesFromASource> receives;
	std::vector<receivesFromASource>::iterator receivesIterator;
	std::size_t receivesCounter = 0;
	std::size_t totalChunksToReceive = 0;

	std::size_t bufferArrayNoOfElements = 0; /*DEBUG*/

	for (int i = 0; i < numOfTasks; i++)
	{
		if (commMatrix[i][taskID] >= 0) // >= 0 is comm
		{
			if (taskID != i) // we don't take into account the elements on a main diagonal, or?
			{
				receives.push_back(receivesFromASource()); // add a blank chunk

				// populate the blank chunk with data
				receives[receivesCounter].senderTaskID = i;
				// Even if both tasks reside on the same HPU, the data is sent - behavior of the real apps
				receives[receivesCounter].hpuOfSenderTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);

				// Separating decimal and whole part of the communication amount
				// 1. Temp variables to store int and decimal parts (all have to be double)
				double wholePartOfComm, decimalPartOfComm, temp;
				// 2. Splitting the comm
				decimalPartOfComm = modf (commMatrix[i][taskID], &wholePartOfComm);
				// 3. Storing the int part
				receives[receivesCounter].numOfWholeChunks = wholePartOfComm;
				// 4. Multiplying the decimal part with the chunk size (to get that part of the 10MB chunk), and saving only the int part of the result (the MPI_Recv cannot have a decimal count parameter), temp is not used later on
				temp = modf ((decimalPartOfComm * COMM_CHUNK_SIZE), &wholePartOfComm);
				// 5. Storing the partial comm chunk size (number of ints to recv)
				receives[receivesCounter].partialChunkSize = wholePartOfComm;

				totalChunksToReceive += receives[receivesCounter].numOfWholeChunks;

				bufferArrayNoOfElements += (receives[receivesCounter].numOfWholeChunks * COMM_CHUNK_SIZE) + receives[receivesCounter].partialChunkSize; /*DEBUG*/

				if (receives[receivesCounter].partialChunkSize > 0)
				{
					receives[receivesCounter].hasPartialChunk = true;
					totalChunksToReceive++;
				}

				std::cout << "\t\t\t\tRECV: T" << taskID << "(P" << this->getProcessID() << ") from T" << i << "(P" << receives[receivesCounter].hpuOfSenderTask << "): chunks: " << receives[receivesCounter].numOfWholeChunks << " + " << ((wholePartOfComm > 0) ? 1 : 0) << "\n";
				receivesCounter++;
			}
		}
	}

	std::cout << "\t\t\t\tTask " << taskID << " recv buffer: " << bufferArrayNoOfElements << ", size in MB: " << ((bufferArrayNoOfElements * 4)/1024)/1024 << std::endl;

	MPI_Request *commRecvRequests = new MPI_Request[totalChunksToReceive]; // N receives for whole parts + 1 receive for decimal part (if it exists)

	// the total number of chunks received from all sources (necessary to store requests in a global array of receives for this task)
	int chunkCounter = 0;

	receivesCounter = 0; // reset for reuse

	// HTML: record the start time of all receives
	ssStart << getTimestamp(stamp) * htmlFileTimeScalar;

	// process each source that sends data to this task
	for (receivesIterator = receives.begin(); receivesIterator != receives.end(); ++receivesIterator)
	{
		printTimestamp(stamp);
		std::cout << "\tT" << taskID << "\tstart receive from T" << receivesIterator->senderTaskID << std::endl;

		// whole part of comm amount (10MB chunks)
		for (std::size_t j = 0; j < receivesIterator->numOfWholeChunks; j++)
		{
			MPI_Irecv(commDataChunkRecv, COMM_CHUNK_SIZE, MPI_INT, receivesIterator->hpuOfSenderTask, MPI_ANY_TAG, MPI_COMM_WORLD, &commRecvRequests[chunkCounter]);
//			outfile << "RECV: T" << taskID << "(P" << this->getProcessID() << ") from T" << receivesIterator->senderTaskID << "(P" << receivesIterator->hpuOfSenderTask << "): TAG: ANY \n";
			chunkCounter++;
		}

		// decimal part of comm amount (part of a 10MB chunk)
		if (receivesIterator->hasPartialChunk)
		{
			MPI_Irecv(commDataChunkRecv, COMM_CHUNK_SIZE, MPI_INT, receivesIterator->hpuOfSenderTask, MPI_ANY_TAG, MPI_COMM_WORLD, &commRecvRequests[chunkCounter]);
//			outfile << "RECV: T" << taskID << "(P" << this->getProcessID() << ") from T" << receivesIterator->senderTaskID << "(P" << receivesIterator->hpuOfSenderTask << "): TAG: ANY \n";
			chunkCounter++;
		}

		receivesCounter++;

	} // all receives posted

	// wait for receives
	printTimestamp(stamp);
	std::cout << "\tT" << taskID << "\treceives posted.\n";

	MPI_Waitall(chunkCounter, commRecvRequests, MPI_STATUSES_IGNORE);

	// release memory
	delete[] commRecvRequests;

	// HTML: record the stop time of all receives and output everything to HTML
	ssStop << getTimestamp(stamp) * htmlFileTimeScalar;
	htmlFileString += "[ 'P" + ssProcessID.str() + "', 'T" + ssTaskID.str() + " - Receive', '#E5E7E9', new Date(" + ssStart.str() + "), new Date(" + ssStop.str() + ") ],\n";

	printTimestamp(stamp);
	std::cout << "\tT" << taskID << "\tfinished all receives." << std::endl;

}

/**** Send unlock/comm
 * Uses SINGLE 10MB buffer which is reused constantly
 * Send every successor a signal that his precedence is unlocked by a task calling this function, or send data to it
 * Checks column of an alloc matrix, if a value is 0 or greater this means that task depends
 * on the task corresponding to the row number,
 * then receive prec from the PROCESS assigned to the precedence task
 * ****/
void Task::sendCommDAGSingleBuffers(double **commMatrix, int **schedule, int *commDataChunkSend, MPI_Request *sendRequests, std::size_t &sendRequestsCounter, double stamp)
{
	int taskID = this->getID();

	// initially create an array of data about chunks sent to each destination task, one instance of struct per source task
	struct sendsToADestination
	{
		std::size_t receiverTaskID;
		std::size_t hpuOfReceiveTask;
		std::size_t numOfWholeChunks;
		std::size_t partialChunkSize = 0;
		bool hasPartialChunk = false;
	};
	std::vector<sendsToADestination> sends;
	std::vector<sendsToADestination>::iterator sendsIterator;
	std::size_t sendsCounter = 0;
	std::size_t totalChunksToSend = 0;

	std::size_t bufferArrayNoOfElements = 0; /*DEBUG*/

	for (int i = 0; i < numOfTasks; i++)
	{
		if (commMatrix[taskID][i] >= 0) // >= 0 is comm
		{
			if (taskID != i) // we don't take into account the elements on a main diagonal, or?
			{
				sends.push_back(sendsToADestination()); // add a blank chunk

				// populate the blank chunk with data
				sends[sendsCounter].receiverTaskID = i;
				// Even if both tasks reside on the same HPU, the data is sent - behavior of the real apps
				sends[sendsCounter].hpuOfReceiveTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);

				// Separating decimal and whole part of the communication amount
				// 1. Temp variables to store int and decimal parts (all have to be double)
				double wholePartOfComm, decimalPartOfComm, temp;
				// 2. Splitting the comm
				decimalPartOfComm = modf (commMatrix[taskID][i], &wholePartOfComm);
				// 3. Storing the int part
				sends[sendsCounter].numOfWholeChunks = wholePartOfComm;
				// 4. Multiplying the decimal part with the chunk size (to get that part of the 10MB chunk), and saving only the int part of the result (the MPI_Send cannot have a decimal count parameter), temp is not used later on
				temp = modf ((decimalPartOfComm * COMM_CHUNK_SIZE), &wholePartOfComm);
				// 5. Storing the partial comm chunk size (number of ints to send)
				sends[sendsCounter].partialChunkSize = wholePartOfComm;

				totalChunksToSend += sends[sendsCounter].numOfWholeChunks;

				bufferArrayNoOfElements += (sends[sendsCounter].numOfWholeChunks * COMM_CHUNK_SIZE) + sends[sendsCounter].partialChunkSize; /*DEBUG*/

				if ( sends[sendsCounter].partialChunkSize > 0)
				{
					sends[sendsCounter].hasPartialChunk = true;
					totalChunksToSend++;
				}

				sendsCounter++;
			}
		}
	}

	std::cout << "\t\t\t\tTask " << taskID << " send buffer: " << bufferArrayNoOfElements << ", size in MB: " << ((bufferArrayNoOfElements * 4)/1024)/1024 << std::endl;

	// the total number of chunks received from all sources (necessary to store requests in a global array of receives for this task)
	int chunkCounter = 0;

	// process each source that sends data to this task
	for (sendsIterator = sends.begin(); sendsIterator != sends.end(); ++sendsIterator)
	{
		printTimestamp(stamp);
/*DEBUG*/	std::cout << "\tT" << taskID << "\tstart send to T" << sendsIterator->receiverTaskID << std::endl;

		// whole part of comm amount (10MB chunks)
		for (std::size_t j = 0; j < sendsIterator->numOfWholeChunks; j++)
		{
			MPI_Isend(commDataChunkSend, COMM_CHUNK_SIZE, MPI_INT, sendsIterator->hpuOfReceiveTask, chunkCounter, MPI_COMM_WORLD, &sendRequests[sendRequestsCounter]);
			//outfile << "SEND: T" << taskID << "(P" << this->getProcessID() << ") to T" << sendsIterator->receiverTaskID << "(P" << sendsIterator->hpuOfReceiveTask << "): TAG: " << chunkCounter << "\n";
			chunkCounter++;
			sendRequestsCounter++;
		}

		// decimal part of comm amount (part of a 10MB chunk)
		if (sendsIterator->hasPartialChunk)
		{
			MPI_Isend(commDataChunkSend, sendsIterator->partialChunkSize, MPI_INT, sendsIterator->hpuOfReceiveTask, chunkCounter, MPI_COMM_WORLD, &sendRequests[sendRequestsCounter]);
			//outfile << "PSEND: T" << taskID << "(P" << this->getProcessID() << ") to T" << sendsIterator->receiverTaskID << "(P" << sendsIterator->hpuOfReceiveTask << "): TAG: " << chunkCounter << "\n";
			chunkCounter++;
			sendRequestsCounter++;
		}

	} // all sends posted

	// wait for sends
	printTimestamp(stamp);
	std::cout << "\tT" << taskID << "\tsends posted.\n";

//	int flag;
//    MPI_Testall(chunkCounter, commSendRequests, &flag, MPI_STATUSES_IGNORE);
//    while (!flag)
//    {
//    	for (int i = 0; i < chunkCounter; i++)
//    	{
//    		int flag;
//    		MPI_Status status;
//    		MPI_Test(&(commSendRequests[i]), &flag, &status);
//    		outfile << taskID << "\t" << this->getProcessID() << "\t" << status.MPI_SOURCE << "\t" << status.MPI_TAG << "\t" << ((flag) ? "1" : "0") << std::endl;
//    	}
//    	sleep(1);
//        MPI_Testall(chunkCounter, commSendRequests, &flag, MPI_STATUSES_IGNORE);
//    }

	// It is irrelevant to store info about sends!
	// Output to the log file
	printTimestamp(stamp);
	std::cout << "\tT" << taskID << "\tfinished all sends." << std::endl;

//	int dummy = 0; //variable sent as a precedence semaphore
//	int taskID = this->getID();
//	for (int i = 0; i < numOfTasks; i++)
//	{
//		if (commMatrix[taskID][i] == 0)
//		{
//			printTimestamp(stamp);
//			int hpuOfChildTask = findHPUForTaskInSchedule(schedule, numOfHPUs, maxNumOfTasksInSchedule, i);
///*DEBUG*/	std::cout << "\tP" << ":\tT" << taskID << ": unlocking T" << i << " (P" << hpuOfChildTask << ")" << std::endl;
//			if (this->getProcessID() != hpuOfChildTask) // send unlock only to another process (not to send to itself)
//				MPI_Isend(&dummy, 1, MPI_INT, hpuOfChildTask, i, MPI_COMM_WORLD, &request);
//			printTimestamp(stamp);
///*DEBUG*/std::cout << "\tP" << this->getProcessID() << ":\tT" << taskID << ": unlocked T" << i << " (P" << hpuOfChildTask << ")" << std::endl;
//		}
//	}
}


/****Task computation phase, amount and type of the computation are inputs ****/
void Task::compute(int **compAmounts, double stamp, std::string &htmlFileString)
{
	// Stringstreams for populating the HTML output
	std::stringstream ssProcessID, ssTaskID, ssStart, ssStop;
	ssProcessID << this->getProcessID();
	ssTaskID << this->getID();
	ssStart << getTimestamp(stamp) * htmlFileTimeScalar;

	printTimestamp(stamp);

	// Store the computation information (and its start time) about this task
	std::cout << "\tT" << this->getID() << "\tcomp amounts";
	for (int i = 0; i < numOfCompTypes; i++)
		std::cout << "\t" << compAmounts[this->getID()][i] * compAmountMultiplicator;
	std::cout << " started." << std::endl;

	// Execute computations
	for (int i = 0; i < numOfCompTypes; i++)
	{
		for (int j = 0; j < compAmounts[this->getID()][i] * compAmountMultiplicator; j++)
			execute_load(i);
	}

	// Output to HTML
	ssStop << getTimestamp(stamp) * htmlFileTimeScalar;
	htmlFileString += "[ 'P" + ssProcessID.str() + "', 'T" + ssTaskID.str() + " - Work', '#797D7F', new Date(" + ssStart.str() + "), new Date(" + ssStop.str() + ") ],\n";

	// Store the computation information (and its stop time) about this task
	printTimestamp(stamp);
	std::cout << "\tT" << this->getID() << "\tcomp amounts";
	for (int i = 0; i < numOfCompTypes; i++)
		std::cout << "\t" << compAmounts[this->getID()][i] * compAmountMultiplicator;
	std::cout << " finished." << std::endl;
}

/****Synthetic load, load type is the input
 * type 0 - 2 x 10^8 short operations (STILL CHECK)
 * type 1 - 2 x 10^8 integer operations
 * type 2 - 2 x 10^8 float operations
 * type 3 - 2 x 10^8 double operations
 * ****/
void Task::execute_load(int comp_type)
{
	short temps1 = 1, temps2 = 2, temps3; //short entites (still check)
	int tempi1 = 1, tempi2 = 2, tempi3; //int entities
	float tempf1 = 1.00, tempf2 = 2.00, tempf3; // float entities
	double tempd1 = 1.00, tempd2 = 2.00, tempd3; // double entities
	if (comp_type == 0)
	{
		for (int i=0; i<COMP_ITERATIONS_SINGLE_LOOP; i++)
		{
			for (int j=0; j<COMP_ITERATIONS_SINGLE_LOOP; j++)
			{
				temps3 += temps1 + temps2;
			}
		}
	}
	else if (comp_type == 1)
	{
		for (int i=0; i<COMP_ITERATIONS_SINGLE_LOOP; i++)
		{
			for (int j=0; j<COMP_ITERATIONS_SINGLE_LOOP; j++)
			{
				tempi3 += tempi1 + tempi2;
			}
		}
	}
	else if (comp_type == 2)
	{
		for (int i=0; i<COMP_ITERATIONS_SINGLE_LOOP; i++)
		{
			for (int j=0; j<COMP_ITERATIONS_SINGLE_LOOP; j++)
			{
				tempf3 += tempf1 + tempf2;
			}
		}
	}
	else
	{
		for (int i=0; i<COMP_ITERATIONS_SINGLE_LOOP; i++)
		{
			for (int j=0; j<COMP_ITERATIONS_SINGLE_LOOP; j++)
			{
				tempd3 += tempd1 + tempd2;
			}
		}
	}
}

/**** GET A HPU ID WHICH RUNS A GIVEN TASK
 *
 * ****/
int Task::findHPUForTaskInSchedule(int **schedule, std::size_t numOfHPUs, std::size_t maxNumTasksInSchedule, int taskID)
{
	for (std::size_t i = 0; i < numOfHPUs; i++)
	{
		for (std::size_t j = 0; j < maxNumTasksInSchedule; j++)
		{
			if ( schedule[i][j] == taskID ) return i;
		}
	}
	return -1;
}


/****Load data phase of a task [FUTURE UPGRADE]
 * amount - the amount of data from file (large dummy file)
 * ****/
void Task::load_data (int amount)
{
}

/****Store data phase of a task [FUTURE UPGRADE]
 * amount - the amount of data from file (large dummy file)
 * ****/
void Task::store_data (int amount)
{
}

#endif /* TASK_H_ */
