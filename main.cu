#include "nBodyCuda.cu"

using namespace std;

int main(int argc, char *argv[]){

			///********** Simulation Variables ***********///

	char * dataName;
	unsigned int numPoints;
	int numSteps;
	int saveStep = 5;
	int coolStep = 500;
	int sleepTime = 15;
	float dt = 0.001;
	float softSquared = 0.0001;

			///********** Apply CLI **********///

	if(argc == 1 || strcmp(argv[1], "help") == 0){
		cout << "\n	Usage: \n" << endl
			 << "	-i [input file (req)] " << endl
			 << "	-n [numPoints (req)] " << endl
			 << "	-s [numSteps (req)] " << endl
			 << "	-e [saveSteps (%5)] " << endl
			 << "	-c [coolStep (500)] " << endl
			 << "	-b [breakTime (15)] " << endl
			 << "	-t [stepSize (0.001)] " << endl;
		return 0;
	}

	extern char* optarg;
	int opt;
	bool inputSet = false; bool numSet = false; bool stepsSet = false;
	while ((opt = getopt(argc, argv, "i:n:s:e:c:b:t:")) != -1){
		switch(opt) {
			case 'i':
			{
				dataName = optarg; 
				inputSet = true;
				break;
			}
			case 'n':
				numPoints = atoi(optarg);
				numSet = true;
				break;
			case 's':
				numSteps = atoi(optarg);
				stepsSet = true;
				break;
			case 'e':
				saveStep = atoi(optarg);
				break;
			case 'c':
				coolStep = atoi(optarg);
				break;
			case 'b':
				sleepTime = atoi(optarg);
				break;
			case 't':
				dt = stof(optarg);
				break;
			}
		}
	if(inputSet==false||numSet==false||stepsSet==false){
		cout << "	Missing required inputs" << endl;
		return 0;
	}

	cout << "\n\n	Using Parameters: \n"
		 << "	Input Data: " << dataName << endl
		 << "	Points: " << numPoints << endl
		 << "	Steps: " << numSteps << endl
		 << "	dt: " << dt << endl
		 << "	Saving Every: " << saveStep << " steps " << endl
		 << "	Pausing to cool every: " << coolStep << " numSteps " << endl
		 << "	For " << sleepTime << " seconds " << endl << endl;


			///********** N-Body Simulation, Self Cleaning **********///

	nBodySim(numPoints, numSteps, saveStep, coolStep, sleepTime,
			 dt, softSquared, dataName);

			///********** Save Parameters for Animation **********///

	FILE * paramFile = fopen("params", "w");
	fprintf(paramFile, "%i%s%i%s", numPoints, "\t", (int)numSteps/saveStep, "\t");
	fclose(paramFile);
}
