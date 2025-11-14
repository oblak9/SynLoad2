/*
 * utility.h
 *
 *  Created on: Mar 5, 2014
 *      Author: zdravko
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include <sstream>
#include <string>
#include "task.h"

//using namespace std;

std::size_t numOfTasks = 0;
std::size_t numOfCompTypes = 0; // 4 for now
std::size_t numOfHPUs = 0;
std::size_t maxNumOfTasksInSchedule = 0;

// MISC
int compAmountMultiplicator = 1; // Default is 1, used to increase the amount of comp (if too low compared to comm time)

// WINDOWS ENV
std::string inputFileCommMatrix = "C:\\Users\\krpic\\Dropbox\\workspace\\SynLoad2\\input\\SynLoadTaskCommMatrix.txt";
std::string inputFileCompAmounts = "C:\\Users\\krpic\\Dropbox\\workspace\\SynLoad2\\input\\SynLoadTaskCompAmounts.txt";
std::string inputFileTaskOrder = "C:\\Users\\krpic\\Dropbox\\workspace\\SynLoad2\\input\\SynLoadTaskOrder.txt";
std::string outputFilename = "C:\\Users\\krpic\\Dropbox\\workspace\\SynLoad2\\output\\OUT_P";
std::string outputHTMLFilename = "C:\\Users\\krpic\\Dropbox\\workspace\\SynLoad2\\output\\Timeline.html";

//// LINUX ENV
//std::string inputFileCommMatrix = "/home/zdravak/clustershared/synload/input/SynLoadTaskCommMatrix.txt";
//std::string inputFileCompAmounts = "/home/zdravak/clustershared/synload/input/SynLoadTaskCompAmounts.txt";
//std::string inputFileTaskOrder = "/home/zdravak/clustershared/synload/input/SynLoadTaskOrder.txt";
//std::string outputFilename = "/home/zdravak/clustershared/synload/output/OUT_P";
//std::string outputHTMLFilename = "/home/zdravak/clustershared/synload/output/Timeline.html";

/**** Header    and footer for the timeline visualization via the HTML file (the same exists in both Partitioning and Scheduling projects) ****/
// The multiplicator for the time intervals that are stored to the HTML file (values can be really low, then the timeline ticks are not displayed, therefore, multiply all of them with this)
double htmlFileTimeScalar = 1000;
std::string htmlFileHeader = "<html>\n\
  <head>\n\
    <script type=\"text/javascript\" src=\"https://www.gstatic.com/charts/loader.js\"></script>\n\
    <script type=\"text/javascript\">\n\
      google.charts.load('current', {'packages':['timeline']});\n\
      google.charts.setOnLoadCallback(drawChart);\n\
      function drawChart() {\n\
        var container = document.getElementById('timeline');\n\
        var chart = new google.visualization.Timeline(container);\n\
        var dataTable = new google.visualization.DataTable();\n\
\n\
        dataTable.addColumn({ type: 'string', id: 'Processor' });\n\
        dataTable.addColumn({ type: 'string', id: 'Task' });\n\
		dataTable.addColumn({ type: 'string', id: 'style', role: 'style' });\n\
        dataTable.addColumn({ type: 'date', id: 'Start' });\n\
        dataTable.addColumn({ type: 'date', id: 'End' });\n\
        dataTable.addRows([\n";

std::string htmlFileFooter = "\n]);\n\
        chart.draw(dataTable);\n\
      }\n\
    </script>\n\
  </head>\n\
  <body>\n\
    <div id=\"timeline\" style=\"height: 1000px;\"></div>\n\
  </body>\n\
</html>";

struct sendsInfo
{
	std::size_t numOfChunks;
};

#define COMM_CHUNK_SIZE 2621440
#define COMP_ITERATIONS_SINGLE_LOOP 10000
#define DEBUG_HEAD_NODE true

/****
 * FILENAME_CREATOR creates unique output filename for each process.
 * EXAMPLE: out1.txt, out23.txt
 * PARAMETER: rank - process rank
 * OUTPUTS: filename as a string type
 */
std::string filename_creator(int rank)
{
	std::stringstream sstr;
	std::string c;
	sstr << "out";
	sstr << rank;
	sstr << ".txt";
	c = sstr.str();
	return c;
}

/**** STORE STRING TO A FILE
 *
 */
void storeStringToFile(std::string stringToStore, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str());
	if (file.is_open())
	{
		file << stringToStore;
	}
	else std::cout << "ERROR in opening file " << filename << std::endl;
}

/****
 * GET MPI TIMESTAMP
 */
double getTimestamp (double stamp)
{
	return (MPI_Wtime() - stamp);
}

/****
 * MPI TIMESTAMP prints the current process MPI wallclock.
 */
void printTimestamp (double stamp)
{
	//printf("%.4f", MPI_Wtime() - stamp);
	std::cout << MPI_Wtime() - stamp;
}

/**** INITIALIZE 2D DYNAMIC ARRAY
 *
 */
template<typename t>
void initialize2DArray(t **array, int rows, int columns, t initValue)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			array[i][j] = initValue;
		}
	}
}

/**** INITIALIZE 1D DYNAMIC ARRAY
 *
 */
template<typename t>
void initialize1DArray(t *array, int numOfElements, t initValue)
{
	for (int i = 0; i < numOfElements; i++)
	{
		array[i] = initValue;
	}
}

/**** CHECK IF 2D DYNAMIC ARRAY CONTAINS A CERTAIN ELEMENT
 *
 */
template<typename t>
bool does2DArrayContainElement(t **array, int rows, int columns, t searchValue)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			if (array[i][j] == searchValue) return true;
		}
	}
	return false;
}

/**** CHECK IF 1D DYNAMIC ARRAY CONTAINS A CERTAIN ELEMENT
 *
 */
template<typename t>
bool does1DArrayContainElement(t *array, int numOfElements, t searchValue)
{
	for (int i = 0; i < numOfElements; i++)
	{
		if (array[i] == searchValue) return true;
	}
	return false;
}

/**** CHECK IF 2D DYNAMIC ARRAY CONTAINS AN ELEMENT GREATER THAN SPECIFIC VALUE IN A SPECIFIC ROW OR COLUMN
 *
 */
template<typename t>
bool does2DArrayContainElementIn1DGreaterThan(t **array, int rowColumnID, int width, int height, t searchValue, bool row)
{
	if (row)
	{
		for (int i = 0; i < width; i++)
		{
			if (array[rowColumnID][i] > searchValue) return true;
		}
	}
	else
	{
		for (int i = 0; i < height; i++)
		{
			if (array[i][rowColumnID] > searchValue) return true;
		}
	}
	return false;
}


/**** PRINT THE 2D DYNAMIC ARRAY
 *
 */
template<typename t>
void print2DArray(t **array, int rows, int columns)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			std::cout << array[i][j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

/**** PRINT THE 1D DYNAMIC ARRAY
 *
 */
template<typename t>
void print1DArray(t *array, int numOfElements)
{
	for (int i = 0; i < numOfElements; i++)
	{
		std::cout << array[i] << "\t";
	}
	std::cout << std::endl;
}

/**** PACK 2D DATA TO 1D DATA
 *
 */
template<typename t>
void pack2Dto1DArray(t **array2D, t *array1D, int rows, int columns)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			array1D[(i * columns) + j] = array2D[i][j];
		}
	}
}

/**** PACK 2D DATA TO 1D DATA
 *
 */
template<typename t>
void unpack1Dto2DArray(t **array2D, t *array1D, int rows, int columns)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			array2D[i][j] = array1D[(i * columns) + j];
		}
	}
}



#endif /* UTILITY_H_ */
