

// includes, system
#include <iostream>
#include <stdlib.h>

#include <boost/lexical_cast.hpp>

static void show_usage(std::string name)
{
	std::cerr << "Usage: " << name << std::endl
		<< "Options:\n"
		<< "\t-h,\tShow this help message\n"
		<< "\t-p,\tGraph parameters\n" 
		<< "\t\t\tNumber of Nodes (int)\n"
		<< "\t\t\tGraph density(0.0-1.0)\n"
		<< "\t\t\tLowest weight (int)\n"
		<< "\t\t\tHighest weight (int)\n"
		<< std::endl;
}


int
main(int argc, char **argv)
{

	using boost::lexical_cast;
	using boost::bad_lexical_cast;


	int numNodes = 0;
	float density = 0.0;
	int weightRangeLow = 0;
	int weightRangeUp = 0;

	if (argc == 1 && argv[1] == "-h") {
		show_usage(argv[0]);
	}
	else if (argc != 6) { //first argument always program name, second is -p flag
		std::cout << "Please provide 4 arguments." << std::endl;
		show_usage(argv[0]);
	}
	else if (argc == 6) {
		
		try
		{
			numNodes = lexical_cast<int>(argv[2]);
			density = lexical_cast<float>(argv[3]);
			weightRangeLow = lexical_cast<int>(argv[4]);
			weightRangeUp = lexical_cast<int>(argv[5]);
			std::cout << "Creating graph with parameters " << std::endl
				<< "Number of Nodes: " << numNodes << std::endl
				<< "Graph density " << density << std::endl
				<< "Weight range [" << weightRangeLow << "," << weightRangeUp << "] " << std::endl;
				 

		}
		catch (bad_lexical_cast&) {
			std::cerr << "Can not parse graph parameters!" << std::endl;
			exit(EXIT_FAILURE);
		}

	}


	// create graph 

	exit(EXIT_SUCCESS);


}
