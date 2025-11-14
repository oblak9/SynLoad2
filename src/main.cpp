#include <stdlib.h>
#include <iostream>
#include <fstream> // for redireting output to files
#include <time.h>
#include <mpi.h>
#include <string> // for filename modifications
#include <iomanip> // for setting the number of decimals in stdout-ing doubles
#include "task.h"
#include "utility.h"
#include "data.h"

//using namespace std;

/****
 * MAIN FUNCTION
 * ****/
int main (int argc, char **argv)
{

	// for setting the number of decimals in stdout
	std::cout << std::fixed;
	std::cout << std::setprecision(7);

	// MPI settings and basic info
    MPI_Init(&argc, &argv);
    int size, rank, len;
	MPI_Comm_size(MPI_COMM_WORLD, &size); // comm size
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // comm rank

	// Visualization via HTML: create the file string, master immediately adds the header to it
	std::string htmlFileString = "";
	if (rank == 0) htmlFileString = htmlFileHeader;
	std::stringstream ss;
	ss << rank;
	htmlFileString += "[ 'P" + ss.str() + "', '-', '#E5E7E9', new Date(0), new Date(0) ],\n";

	// Redirect all output into files
	std::string filename = outputFilename + std::to_string(rank) + ".txt";
	std::ofstream out(filename);
	std::streambuf *coutbuf = std::cout.rdbuf();
	std::cout.rdbuf(out.rdbuf());

	// MASTER PROCESS reads the initial data (4 parameters) from the files && sends them to slaves
	// WARNING: the files in Linux and Windows differ in the newline character - Windows has 2 (\r\n or CRLF), Linux has one (\n or LF)
	// 			In Linux use unix2dos to convert input txt files
	if (rank == 0)
	{
		getParamsFromTxtFiles(inputFileCompAmounts, inputFileTaskOrder, numOfTasks, numOfHPUs, numOfCompTypes, maxNumOfTasksInSchedule);

		// send data to slaves
		for (int i = 1; i < size; i++)
		{
			MPI_Send(&numOfTasks, 1, MPI_INT, i, i, MPI_COMM_WORLD);
			MPI_Send(&numOfHPUs, 1, MPI_INT, i, i, MPI_COMM_WORLD);
			MPI_Send(&numOfCompTypes, 1, MPI_INT, i, i, MPI_COMM_WORLD);
			MPI_Send(&maxNumOfTasksInSchedule, 1, MPI_INT, i, i, MPI_COMM_WORLD);
		}
	}

	// SLAVE PROCESSES receive the initial data
	else
	{
		MPI_Recv(&numOfTasks, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&numOfHPUs, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&numOfCompTypes, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&maxNumOfTasksInSchedule, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	std::cout 	<< "*****************************\n"
				<< "Reporting parameters: \t numOfTasks \t numOfHPUs \t numOfCompTypes \t maxNumOfTasksInSchedule\n"
				<< "Reporting parameters: \t" << numOfTasks << "\t" << numOfHPUs << "\t" << numOfCompTypes << "\t" << maxNumOfTasksInSchedule << std::endl;

	// MASTER PROCESS checks if the program was run using the required number of processes
	// If not, the MPI is finalized and the main() returns 0, thus the programs stops
	if (size != numOfHPUs)
	{
		if (rank == 0)
		{
			std::cout << "The program was not run on the correct amount of processes!\nRequired processes: " << numOfHPUs << "\nProcesses used: " << size << "\n";
		}
		MPI_Finalize();
		return 1; // 1 indicates error it seems
	}

	// ALL PROCESSES create & initialize data containers
	int **compAmounts;
	compAmounts = new int*[numOfTasks];
	for (std::size_t i = 0; i < numOfTasks; i++)
		compAmounts[i] = new int[numOfCompTypes];
	initialize2DArray(compAmounts, numOfTasks, numOfCompTypes, 0);

	int **schedule;
	schedule = new int*[numOfHPUs];
	for (std::size_t i = 0; i < numOfHPUs; i++)
		schedule[i] = new int[maxNumOfTasksInSchedule];
	initialize2DArray(schedule, numOfHPUs, maxNumOfTasksInSchedule, -1);

	double **commMatrix;
	commMatrix = new double*[numOfTasks];
	for (std::size_t i = 0; i < numOfTasks; i++)
		commMatrix[i] = new double[numOfTasks];
	initialize2DArray(commMatrix, numOfTasks, numOfTasks, -1.0);

	// Create temporary data containers (packed to 1D arrays) to send via MPI (all processes)
	int *packedCompAmounts;
	packedCompAmounts = new int[numOfTasks * numOfCompTypes];

	int *packedSchedule; // = (int *)malloc( (numOfHPUs * maxNumOfTasksInSchedule) * sizeof(int));
	packedSchedule = new int[numOfHPUs * maxNumOfTasksInSchedule];

	double *packedCommMatrix;
	packedCommMatrix = new double[numOfTasks * numOfTasks];

	// MASTER PROCESS loads containers with data from files && sends data to slaves
	if (rank == 0)
	{
		getCompAmountsFromTxtFile(inputFileCompAmounts, compAmounts);
		getScheduleFromTxtFile(inputFileTaskOrder, schedule);
		getCommMatrixFromTxtFile(inputFileCommMatrix, commMatrix);

		// pack to 1D arrays (to have contiguous memory, otherwise MPI comm does not work)
		pack2Dto1DArray(compAmounts, packedCompAmounts, numOfTasks, numOfCompTypes);
		pack2Dto1DArray(schedule, packedSchedule, numOfHPUs, maxNumOfTasksInSchedule);
		pack2Dto1DArray(commMatrix, packedCommMatrix, numOfTasks, numOfTasks);

		// send data to slaves
		for (int i = 1; i < size; i++)
		{
			MPI_Send(packedCompAmounts, (numOfTasks * numOfCompTypes), MPI_INT, i, i, MPI_COMM_WORLD);
			MPI_Send(packedSchedule, (numOfHPUs * maxNumOfTasksInSchedule), MPI_INT, i, i, MPI_COMM_WORLD);
			MPI_Send(packedCommMatrix, (numOfTasks * numOfTasks), MPI_DOUBLE, i, i, MPI_COMM_WORLD);
		}
	}
	// SLAVE PROCESSES receive and unpack data
	else
	{
		MPI_Recv(packedCompAmounts, (numOfTasks * numOfCompTypes), MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(packedSchedule, (numOfHPUs * maxNumOfTasksInSchedule), MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(packedCommMatrix, (numOfTasks * numOfTasks), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// unpack data
		unpack1Dto2DArray(compAmounts, packedCompAmounts, numOfTasks, numOfCompTypes);
		unpack1Dto2DArray(schedule, packedSchedule, numOfHPUs, maxNumOfTasksInSchedule);
		unpack1Dto2DArray(commMatrix, packedCommMatrix, numOfTasks, numOfTasks);
	}

	// ALL PROCESSES delete temp packed containers
	/*delete[] packedCompAmounts;
	delete[] packedSchedule;
	delete[] packedCommMatrix;*/

//	// printouts
//	if (rank != 0)
//	{
//		print2DArray(compAmounts, numOfTasks, numOfCompTypes);
//		print2DArray(schedule, numOfHPUs, maxNumOfTasksInSchedule);
//		print2DArray(commMatrix, numOfTasks, numOfTasks);
//	}


	// create send/receive buffers for all processes
	// necessary to have the place from where the data will be sent and where is going to be received (to simulate comm)
	int* commDataChunkSend = new int[COMM_CHUNK_SIZE]; // 10MB of data to use as send container by each process, no need to populate
	int* commDataChunkRecv = new int[COMM_CHUNK_SIZE]; // 10MB of data to use as recv container by each process, no need to populate

	// Process info
	int numOfAllocatedTasks; // number of allocated tasks to a process
	double stamp; // variable to use as a timestamp
    char processorName[MPI_MAX_PROCESSOR_NAME]; // container for processor names TODO: is this correct?

	/**** EVERY RANK does the same
	 * searches for the number of tasks allocated to it
	 * creates a vector to store task IDs dedicated to it
	 * fills that vector
	 * creates actual tasks (objects)
	 * (optional) do benchmarks
	 * runs procedures for every task
	 * ****/
    numOfAllocatedTasks = calculateNumOfAllocTasks(rank, schedule); // get number of tasks allocated to this process

    std::cout << "Num of allocated tasks to this HPU: " << numOfAllocatedTasks << std::endl;

	int *members = new int[numOfAllocatedTasks]; // create an empty vector of task ids allocated to this process
	// TODO: Redo this, since there are no unlocks in DAGs
	MPI_Request *requests = new MPI_Request[numOfAllocatedTasks]; // create an empty vector of MPI requests for nonblocking send (unlocks)

	allocateTasks(members, rank, schedule);

	std::cout << "Tasks allocated to this HPU: ";
	print1DArray(members, numOfAllocatedTasks);

	// create an array of task objects allocated to this process
	Task* taskArray = new Task[numOfAllocatedTasks];

	MPI_Barrier(MPI_COMM_WORLD); //barrier to sync timestamps
	stamp = MPI_Wtime(); // reference time

	/**** Output header for each process
	 * processor name
	 * timestamp
	 */
	MPI_Get_processor_name(processorName, &len); // store the processor name
	std::cout << "Node name: " << processorName << std::endl; //output of the processor name
	printTimestamp(stamp); //initial timestamp for each process
	std::cout << "\tSTART\n*****************************\n";

	// for every task on this process get the number of sends (calls to send function)
	// get the total number of sends on a single process (adding the number of sends that happen on all tasks running on this process/HPU)
	std::size_t numOfSends = 0;
	for (int i=0; i<numOfAllocatedTasks; i++)
	{
		numOfSends += taskArray[i].getTotalNumOfSendChunks(commMatrix);
	}

	// allocate enough requests to control all sends at once
	MPI_Request *processSendRequests = new MPI_Request[numOfSends];
	std::size_t processSendRequestsCounter = 0;

	// for every task run the usual
	for (int i=0; i<numOfAllocatedTasks; i++)
	{
		runSchedule(rank, taskArray[i], i, members, processSendRequests, processSendRequestsCounter, commMatrix, schedule, compAmounts, commDataChunkSend, commDataChunkRecv, stamp, htmlFileString);
	}

	MPI_Waitall(processSendRequestsCounter, processSendRequests, MPI_STATUSES_IGNORE);

	// Visualization via HTML: Master collects the pieces of the file and combines them together, then it adds the footer
	if (rank == 0)
	{
		// Receive from all the processes
		for (int i = 1; i < size; i++)
		{
			MPI_Status status;
			MPI_Probe(i, i, MPI_COMM_WORLD, &status);
			int lengthOfReceivedString;
			MPI_Get_count(&status, MPI_CHAR, &lengthOfReceivedString);
			char *tempHtmlFileChunk = new char[lengthOfReceivedString];
			MPI_Recv(tempHtmlFileChunk, lengthOfReceivedString, MPI_CHAR, i, i, MPI_COMM_WORLD, &status);
			htmlFileString += std::string(tempHtmlFileChunk, lengthOfReceivedString);
			delete [] tempHtmlFileChunk;
		}
		htmlFileString += htmlFileFooter;
	}
	else
	{
		MPI_Send(htmlFileString.c_str(), htmlFileString.length(), MPI_CHAR, 0, rank, MPI_COMM_WORLD);
	}

	if (rank == 0) storeStringToFile(htmlFileString, outputHTMLFilename);


	printTimestamp(stamp);

	/**** DELETING ****/
	for (std::size_t i = 0; i < numOfTasks; i++)
		delete[] compAmounts[i];
	delete[] compAmounts;
	for (std::size_t i = 0; i < numOfHPUs; i++)
		delete[] schedule[i];
	delete[] schedule;
	for (std::size_t i = 0; i < numOfTasks; i++)
		delete[] commMatrix[i];
	delete[] commMatrix;

	delete[] members;

	std::cout << "\tFINISH\n*****************************\n" << std::endl;

	// master writes out the end of the program
	if (rank == 0) printTimestamp(stamp);

	MPI_Finalize();


	/*********************************************/
	/****************INPUT DATA*******************/

	/**** The amount of computation for each task
	 * x 10^8
	 * dimesnionality: numOfTasks x numOfCompTypes
	 * each row is one task
	 * each column is the amount of computation for a computation type
	 * ****/
//	int comp_amount[N] = {6, 7, 2, 4, 6};
//	int comp_amount[numOfTasks] = {5, 3, 2, 3, 6, 7, 7, 9, 3, 7, 9, 1};

	/**** The type of computation for each task
	 * for use in function compute
	 * ****/
	// TODO: obsolete, this data is included in compAmounts
//	int comp_type[N] = {1, 1, 1, 1, 1};
//	int comp_type[numOfTasks] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

	/**** Allocation of tasks to processes
	 * size = num_of_tasks)
	 * process rank = processing unit id
	 * starts from 0, core 1 -> 0
	 * position in this vector is the task id ****/
	// TODO: introducing the order of execution also
	/****
	 * dimesnionality: numOfHPUs x maxNumTasksInSchedule
	 * each row is a schedule for a single HPU
	 * each column is a task in a schedule of a single HPU
	 */
//	int allocation[N] = {2, 1, 0, 2, 2};
//	int allocation[numOfTasks] = {2, 0, 2, 0, 1, 3, 4, 0, 5, 4, 1, 2};


	/****Task interdependency matrix
	 * -1 - no relation (no edge or arc)
	 *  0 - precedence (arc)
	 *  >0 - communication amount (edge)****/
						 //1   2   3   4   5   6   7   8
//
//	int task_dep[N][N] = {-1, -1,  0,  0, -1, //-1, -1, -1, //1
//						  -1, -1, 10, -1, -1, //-1, -1, -1, //2
//						  -1, 10, -1, -1,  0, // 0, -1, -1, //3
//						  -1, -1, -1, -1,  0, //-1, -1,  0, //4
//						  -1, -1, -1, -1, -1 //-1, -1, -1, //5
////						  -1, -1, -1, -1, -1, -1, -1, -1, //6
////						  -1, -1, -1, -1, -1, -1, -1,  0, //7
////						  -1, -1, -1, -1, -1, -1, -1, -1//8
//						 };
//
//	int task_dep[numOfTasks][numOfTasks] =
//	{
//		-1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
//		-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1,
//		-1, -1, -1, -1, 0, 5, -1, -1, -1, -1, -1, -1,
//		-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1,
//		-1, -1, -1, -1, -1, -1, 3, 0, -1, -1, -1, -1,
//		-1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
//		-1, -1, -1, 1, 3, -1, -1, -1, 4, 0, -1, -1,
//		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1,
//		-1, -1, -1, -1, -1, -1, 4, -1, -1, -1, -1, 0,
//		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 0,
//		-1, -1, -1, -1, -1, -1, -1, 10, -1, 8, -1, 0,
//		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
//	};
//

	std::cout.rdbuf(coutbuf); // reset the cout buffer
    return 0;
}
