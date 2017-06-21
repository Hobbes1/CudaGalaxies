#include "nBodyCuda.cu"
#include <math.h>

using namespace std;

const int MAX_GLOB_MEM = 2095251456; 	// total global memory
const int MAX_SHARED_MEM = 49152; 		// shared memory per block
const int MAX_GRID_SIZE = 65535;		// Max 1Dimensional grid size


int main(int argc, char *argv[]){

			///*** NumPoints ***///

	unsigned int numPoints = 16384;

			///*** CUDA Memory Parameters ***///

	const int threadsPerBlock = 512;		// blockSize from NVDA_nbody
	const int numTiles = (numPoints + threadsPerBlock -1) / threadsPerBlock;
	const int sharedMemSize = threadsPerBlock * 2 * sizeof(float4);
	if (numTiles > MAX_GRID_SIZE){
		cout << "Allocated too many tiles" << endl; return 0; }
	if (sharedMemSize > MAX_SHARED_MEM){
		cout << "Allocated too much Shared Memory" << endl; return 0; }
	
			///*** Host Memory Objects ***///

	int steps = 2000;
	int saveStep = 10; 						// save every (N) steps
	int coolStep =  500; 					// pause (sleepTime) every (N) steps
	int sleepTime = 10;
	float dt = 0.001;					// time-steps (in variable time units)
	softSquared = 0.0001;
	cout << "Softening Squared: " << softSquared << endl;
	size_t size4 = numPoints * sizeof(float4);
	size_t size3 = numPoints * sizeof(float3);

	if (3 * size4 + size3 > MAX_GLOB_MEM){
		cout << "Allocated to much Global Memory" << endl; return 0; }

	cout << "allocating shared memory: " << sharedMemSize << " MAX = 49152 " << endl;
	cout << "allocating global memory: " << 3 * size4 + size3 << " MAX = 2095251456" <<endl;
	cout << "allocating grid sizes: " << numTiles << " MAX = 65535" << endl;
	float4* host_points;					// array of float4 point data {m, x, y, z}
	float4* host_velocities;				// initial velocities to be loaded into dev memory
	float3* host_accels;					// accels just used in the GPU

			///*** Device Memory Objects ***///
	
	float4* dev_points;
	float4* dev_velocities;		//examples use velocity float4s 
	float3* dev_accels;			

			///*** File handling ***///

	char filein[20] = {0};
	char fileout[20] = {0};
	char * temp = argv[1];
	strcat(filein, temp);
	strcat(filein, ".data");
	strcat(fileout, temp);
	strcat(fileout, "Run.data");

	FILE *savefile;
	ifstream datafile(filein, ifstream::in);
    savefile = fopen(fileout, "w");
	if(datafile.is_open())
		cout <<"succesfully opened initial and output files"<< endl;

			///*** Load point data from file into _Pinned_ host memory ***///

	std::string line;
	int i = 0; 
	checkCuda( cudaMallocHost((void**)&host_points, size4) ); 		//pinned
	checkCuda( cudaMallocHost((void**)&host_velocities, size4) ); 	//pinned
	checkCuda( cudaMallocHost((void**)&host_accels, size3) );		//pinned
	while(getline(datafile, line)){
		std::stringstream linestream(line);
		if(i==0)
			std::cout << "N = " << numPoints << endl;				 
		else{	
			linestream >> host_points[i-1].w 
					   >> host_points[i-1].x 
					   >> host_points[i-1].y
					   >> host_points[i-1].z 						 //load mass,x,y,z data
					   >> host_velocities[i-1].x
					   >> host_velocities[i-1].y
					   >> host_velocities[i-1].z;			 			 //load vx,vy,vz data

			host_velocities[i-1].w = 0.0;
			host_accels[i-1].x = 0.0;
			host_accels[i-1].y = 0.0;
			host_accels[i-1].z = 0.0;
		}
	i++;
	}

	fprintf(savefile,"%i%s%i%s%12.6f%s",
		    numPoints,"\t",
			steps,"\t",
			dt,"\n");												//header

	checkCuda( cudaMalloc((void**)&dev_points, size4));
	checkCuda( cudaMalloc((void**)&dev_velocities, size4));
	checkCuda( cudaMalloc((void**)&dev_accels, size3));

			///*** Load point data from _Pinned_ into (1 for now) Device memory ***///
	
	checkCuda( cudaMemcpy(dev_points, 
						  host_points, 
						  size4, 
						  cudaMemcpyHostToDevice));					 // points0

	checkCuda( cudaMemcpy(dev_velocities, 
						  host_velocities,
						  size4,
						  cudaMemcpyHostToDevice));					 // velocities
	
	checkCuda( cudaMemcpy(dev_accels,
						  host_accels,
						  size3,
						  cudaMemcpyHostToDevice));					 // accels

			///*** Perform N-Body Integration on GPU, saving data on CPU ***///

	// test that copying back before integrating returns expected values...
	checkCuda( cudaMemcpy(host_points, dev_points, size4, cudaMemcpyDeviceToHost));
	cout << " first point (x) back and forth: " << host_points[0].x << endl;
	// this works and returns the correct (non-NaN) value

	SaveStateB(host_points, numPoints, savefile); //initial save
	int nstep;
	cout << "calling leapstep with: " << numTiles << " gridsize (numblocks), " << threadsPerBlock << " blocksize, and " << sharedMemSize << " shared memory per block" << endl;
	for (nstep = 0; nstep < steps; nstep++){

		cout << "Steps: " << nstep+1 << "\r";
		cout.flush();
		if (nstep % coolStep == 0){
			cout << "		COOLING" << endl;
			sleep(sleepTime);
			cout.flush();
		}

		// the leapstep algorithm in all of its parallel kernel call glory 

		accel_step <<< numTiles, threadsPerBlock, sharedMemSize >>>
				   (dev_points, dev_accels, numPoints, softSquared, dt, numTiles);

		vel_step <<< numTiles, threadsPerBlock, sharedMemSize >>>
				 (dev_velocities, dev_accels, numPoints, dt);

		r_step <<< numTiles, threadsPerBlock, sharedMemSize >>>
			   (dev_points, dev_velocities, numPoints, dt);

		vel_step <<< numTiles, threadsPerBlock, sharedMemSize >>>
				 (dev_velocities, dev_accels, numPoints, dt);


		zero_accels <<< numTiles, threadsPerBlock, sharedMemSize >>>
				   (dev_accels);
	
		if (nstep % saveStep == 0){
			checkCuda( cudaMemcpy(host_points,
								  dev_points,
								  size4,
								  cudaMemcpyDeviceToHost));
			SaveStateB(host_points, numPoints, savefile); 
		}
}
			///*** Free Memory ***///
	cudaFree(dev_points);
	cudaFree(dev_velocities);
	cudaFree(dev_accels);
	cudaFreeHost(host_points);
	cudaFreeHost(host_velocities);
	cudaFreeHost(host_accels);
}
