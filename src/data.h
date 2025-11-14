/*
 * data.h
 *
 *  Created on: 2. stu 2017.
 *      Author: Krpic
 */

#ifndef DATA_H_
#define DATA_H_

#include<string>
#include<fstream>
#include<iterator>
#include<vector>

/**** GET THE SIGNIFICANT PARAMETERS FROM THE FILES
 * Gets the significant parameters from the files, parameters are declared in utility.h and passed by reference in order for this function gives them values
 * @param compAmountsFilename - the name of the input txt file for the comp amounts for each task
 * @param scheduleFilename - the name of the input txt file for the schedule
 * by reference:
 * @param numOfTasks
 * @param numOfHPUs
 * @param numOfCompTypes
 * @param maxNumOfTasksInSchedule
 */
void getParamsFromTxtFiles(std::string compAmountsFilename, std::string scheduleFilename, std::size_t &numOfTasks, std::size_t &numOfHPUs, std::size_t &numOfCompTypes, std::size_t &maxNumOfTasksInSchedule)
{
	std::string stringLine;
	std::ifstream compAmountsFile (compAmountsFilename.c_str());
	if ( compAmountsFile.is_open() )
	{
		while ( std::getline(compAmountsFile, stringLine) )
		{
			// split line into tokens only if this is the first line (all lines have the same number of tokens)
			if (numOfTasks == 0)
			{
				std::stringstream linestream(stringLine);
				std::string token;
				while ( getline(linestream, token, '\t') )
				{
					numOfCompTypes++;
				}
			}

			numOfTasks++;
		}
	}
	else std::cout << "Error in opening file " << compAmountsFilename << std::endl;

	// second file (num of HPUs and the number of tasks in the longest schedule)
	std::ifstream scheduleFile (scheduleFilename.c_str());
	if ( scheduleFile.is_open() )
	{
		while ( std::getline(scheduleFile, stringLine) )
		{
			std::stringstream linestream(stringLine);
			std::string token;
			std::size_t tokenCounter = 0;
			while ( getline(linestream, token, '\t') )
			{
				tokenCounter++;
			}
			if (tokenCounter > maxNumOfTasksInSchedule) maxNumOfTasksInSchedule = tokenCounter;
			numOfHPUs++;
		}
	}
	else std::cout << "Error in opening file " << compAmountsFilename << std::endl;
}

/**** GET COMP AMOUNTS FOR EACH TASK FROM FILE
 * @param filename - the name of the input txt file
 * @param compAmount - 2D array of comp amounts
 */
void getCompAmountsFromTxtFile(std::string filename, int **compAmount)
{
	std::string stringLine;
	std::ifstream file (filename.c_str());
	if ( file.is_open() )
	{
		std::size_t lineCounter = 0;
		while ( std::getline(file, stringLine) )
		{
			// split line into tokens
			std::stringstream linestream(stringLine);
			std::string token;
			std::size_t tokenCounter = 0;
			while( getline(linestream, token, '\t') )
			{
				int intToken = strtol(token.c_str(), NULL, 0);
				compAmount[lineCounter][tokenCounter] = intToken;

				tokenCounter++;
			}
			lineCounter++;
		}
	}
	else std::cout << "Error in opening file " << filename << std::endl;
}

/**** GET SCHEDULE FROM FILE
 * @param filename - the name of the input txt file
 * @param compAmount - 2D array, rows are HPUs, columns are the order of execution of task IDs
 */
void getScheduleFromTxtFile(std::string filename, int **schedule)
{
	std::string stringLine;
	std::ifstream file (filename.c_str());

	// each row has a different length, so we need the biggest one to define a second dimension for the 2D array
	if ( file.is_open() )
	{
		std::size_t lineCounter = 0;
		while ( std::getline(file, stringLine) )
		{
			// split line into tokens
			std::stringstream linestream(stringLine);
			std::string token;
			std::size_t tokenCounter = 0;
			while( getline(linestream, token, '\t') )
			{
				int intToken = strtol(token.c_str(), NULL, 0);
				schedule[lineCounter][tokenCounter] = intToken;

				tokenCounter++;
			}
			lineCounter++;
		}
	}
	else std::cout << "Error in opening file " << filename << std::endl;
}

/**** GET THE COMM MATRIX FROM THE TXT FILE
 * @param filename - the name of the input txt file
 * @param compAmount - comm matrix
 */
void getCommMatrixFromTxtFile(std::string filename, double **commMatrix)
{
	std::string stringLine;
	std::ifstream file (filename.c_str());
	if ( file.is_open() )
	{
		std::size_t lineCounter = 0;
		while ( std::getline(file, stringLine) )
		{
			// split line into tokens
			std::stringstream linestream(stringLine);
			std::string token;
			std::size_t tokenCounter = 0;
			while( getline(linestream, token, '\t') )
			{
				double doubleToken = strtod(token.c_str(), NULL);
				commMatrix[lineCounter][tokenCounter] = doubleToken;

				tokenCounter++;
			}
			lineCounter++;
		}
	}
	else std::cout << "Error in opening file " << filename << std::endl;
}

/****DISPLAY DATA FROM THE MEMBER ATTRIBUTES
 *
 */
void displayData()
{
	std::cout << "Number of tasks: " << numOfTasks << std::endl;
	std::cout << "Number of HPUs: " << numOfHPUs << std::endl;
	std::cout << "Number of comp types: " << numOfCompTypes << std::endl;
	std::cout << "Num of tasks in the longest schedule: " << maxNumOfTasksInSchedule << std::endl;
}

/**** CALCULATE THE NUMBER OF ALLOCATED TASKS TO A CALLING PROCESS AND RETURN THE VALUE
 * needed to create counter for number of tasks created on every process
 * ****/
std::size_t calculateNumOfAllocTasks(int rank, int **schedule)
{
	std::size_t num = 0;
	for (std::size_t i = 0; i < maxNumOfTasksInSchedule; i++)
	{
		if (schedule[rank][i] != -1)
			num++;

	}
	return num;
}


/**** CREATE A VECTOR OF TASK IDs ALLOCATED TO A CALLING PROCESS
 * needed to create ids of tasks created on a process
 * ****/
void allocateTasks(int* members, int rank, int **schedule)
{
	for (std::size_t i = 0; i < maxNumOfTasksInSchedule; i++)
	{
		if (schedule[rank][i] != -1)
			members[i] = schedule[rank][i];
	}
}

/**** STARTS A SCHEDULE FOR THE CALLING PROCESS
 * TODO: Create two versions of it: DAG and MIXED
 * Runs all the necessary functions for each task
 * Takes Task as an input
 * iterator for multiple tasks on one process
 * members vector of tasks on that process
 * task dependency matrix
 * allocation vector
 * amount of computation
 * type of computation
 * initial timestamp
 * ****/
void runSchedule(	int rank,
					Task task,
					int taskCounter,
					int* members,
					MPI_Request *sendRequests,
					std::size_t &sendRequestsCounter,
					double **commMatrix,
					int **schedule,
					int **compAmounts,
					int *commDataChunkSend,
					int *commDataChunkRecv,
					double stamp,
					std::string &htmlFileString)
{
	task.setID(members[taskCounter]);
	// TODO: only for MIXED graph: task.waitUnlocksViaRecv(commMatrix, schedule, stamp);
	// Run receives for a task only if it is not the entry task (i.e. it doesn't receive anything)
	if ( does2DArrayContainElementIn1DGreaterThan(commMatrix, task.getID(), numOfTasks , numOfTasks, -1.0, false) )
		task.recvCommDAGSingleBuffers(commMatrix, schedule, commDataChunkRecv, stamp, htmlFileString);
	task.compute(compAmounts, stamp, htmlFileString);
	// Run sends for a task only if it is not the exit task (i.e. it doesn't send anything)
	if ( does2DArrayContainElementIn1DGreaterThan(commMatrix, task.getID(), numOfTasks , numOfTasks, -1.0, true) )
		task.sendCommDAGSingleBuffers(commMatrix, schedule, commDataChunkSend, sendRequests, sendRequestsCounter, stamp);
	// TODO: only for MIXED graph: task.communicate(task_dep, allocation, commDataChunk, stamp);
	// TODO: only for MIXED graph: task.sendUnlocks(commMatrix, schedule, request, stamp);
}


#endif /* DATA_H_ */
