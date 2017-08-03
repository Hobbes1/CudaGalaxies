#include "nBodyCuda.h" 

<<<<<<< HEAD
		/* nvcc compilation didn't work well with std11 
		 * I was too lazy to figure out why, so I included
		 * a missing function or two */
float stof(const char* s)
{
	float rez = 0, fact = 1;
	if (*s == '-'){
		s++;
		fact = -1;
	};
	for (int point_seen = 0; *s; s++){
		if (*s == '.'){
			point_seen = 1;
			continue;
		};
		int d = *s - '0';
		if (d >= 0 && d <= 9 ){
			if (point_seen) fact /= 10.0f;
			rez = rez * 10.0f + (float)d;
		};
	};
	return rez * fact;
}

=======
>>>>>>> e3977a9aa8cfaf316015eb971540a3df30b629ec
		/* Save Position data to file */
void SaveStateB(float4* bodies, int n, FILE* savefile)
{
		for(int i = 0; i < n; i++)
			fprintf(savefile,"%12.6f%s"
					 "%12.6f%s%12.6f%s",
					 bodies[i].x,"\t",
					 bodies[i].y,"\t",
					 bodies[i].z,"\n");
}

		/* Sometimes I wanted to save acceleration data as well */
void SaveStateA(float4* bodies, float3* accels, int n, FILE* savefile)
{
		float a_r;
		for(int i = 0; i < n; i++){
			a_r = pow((pow(accels[i].x, 2.0) + pow(accels[i].y, 2.0) + 
				  pow(accels[i].z, 2.0)), 0.5);

			fprintf(savefile,"%12.6f%s"
					 "%12.6f%s%12.6f%s"
					 "%12.6f%s",
					 bodies[i].x,"\t",
					 bodies[i].y,"\t",
					 bodies[i].z,"\t",
					 a_r, "\n");
		}
}

		/* Single body-body interaction, sums the acceleration 
		 * quantity across all interactions */
__device__ float3
bodyBodyInteraction(float4 bi, float4 bj, float3 ai, float softSquared) 
{
	float3 r;
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;
	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
	float invDist = rsqrt(distSqr + softSquared);
	float invDistCube = invDist * invDist * invDist;
	float s = bj.w * invDistCube;
	if(r.x != 0.0){
		ai.x += r.x * s;
		ai.y += r.y * s;
		ai.z += r.z * s;
	}
	return ai;
}

		/* Apply body-body interactions in sets of "tiles" as per
		 * NVIDIA's n-body example, loading from shared memory in 
		 * this way speeds the algorithm further */ 
__device__ float3
tile_accel(float4 threadPos, float4 *PosMirror, float3 accel, float softSquared,
		   int numTiles) 
{

	extern __shared__ float4 sharedPos[];

	for (int tile = 0; tile < numTiles; tile++){
		sharedPos[threadIdx.x] = PosMirror[tile * blockDim.x + threadIdx.x];
		__syncthreads();

#pragma unroll 128

		for ( int i = 0; i < blockDim.x; i++ ) {
			accel = bodyBodyInteraction(threadPos, sharedPos[i], accel, softSquared);
		}
		__syncthreads();
	}
	return accel;
}

		/* Acquire all acceleration vectors for the points */ 
__global__ void
accel_step( float4 *__restrict__ devPos,
			float3 *__restrict__ accels,
			unsigned int numBodies,
			float softSquared,
			float dt, int numTiles ) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > numBodies) {return;};
	accels[index] = tile_accel(devPos[index], devPos, accels[index], softSquared, numTiles);
	__syncthreads();
}

		/* Step all point-velocities by 0.5 * a * dt
		 * as per a leapfrog algorithm; is called twice,
		 * once before and once after the position step */
__global__ void
vel_step( float4 *__restrict__ deviceVel,
		  float3 *__restrict__ accels,
		  unsigned int numBodies,
		  float dt)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > numBodies) {return;};
	deviceVel[index].x += accels[index].x * 0.5 * dt;
	deviceVel[index].y += accels[index].y * 0.5 * dt;
	deviceVel[index].z += accels[index].z * 0.5 * dt;
}

		/* Step positions from velocities */
__global__ void
r_step( float4 *__restrict__ devPos,
		float4 *__restrict__ deviceVel,
		unsigned int numBodies,
		float dt)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > numBodies) {return;};
	devPos[index].x += deviceVel[index].x * dt;
	devPos[index].y += deviceVel[index].y * dt;
	devPos[index].z += deviceVel[index].z * dt;
}

		/* Not used */
__global__ void
update_old( float4 *__restrict__ newPos,
			float4 *__restrict__ oldPos ) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	oldPos[index] = newPos[index];
}

		/* zero the acceleration array between leapfrog 
		 * steps, I wasn't sure if a cuda mem-set existed 
		 * and/or would be faster */
__global__ void
zero_accels( float3 *__restrict__ accels ) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	accels[index].x = 0.0f;
	accels[index].y = 0.0f;
	accels[index].z = 0.0f;
}
<<<<<<< HEAD


		/* The big "main" simulation call. Includes all memory
		 * allocation and initial device transfers from datafile
		 * input. Runs the simulation with desired parameters
		 * and cleans up after itself */
void 
nBodySim(unsigned int numPoints,
		 int numSteps,
		 int saveStep,
		 int coolStep,
		 int sleepTime,
		 float dt,
		 float softSquared,
		 char* dataName)
		 
{
	using namespace std;

	char filein[20] = {0};
	char fileout[20] = {0};
	strcat(filein, dataName);
	strcat(filein, ".data");
	strcat(fileout, dataName);
	strcat(fileout, "Run.data");
	std::ifstream datafile(filein, ifstream::in);
    FILE* savefile = fopen(fileout, "w");
	if(datafile.is_open())
		cout <<"	Succesfully opened initial and output files"<< endl;

			///********** CUDA Memory Parameters **********///

	const int threadsPerBlock = 512;		// blockSize from NVDA_nbody
	const int numTiles = (numPoints + threadsPerBlock -1) / threadsPerBlock;
	const int sharedMemSize = threadsPerBlock * 2 * sizeof(float4);
	if (numTiles > MAX_GRID_SIZE){
		cout << "Allocated too many tiles" << endl; return; }
	if (sharedMemSize > MAX_SHARED_MEM){
		cout << "Allocated too much Shared Memory" << endl; return; }
	
			///*** Host Memory Objects ***///

	size_t size4 = numPoints * sizeof(float4);
	size_t size3 = numPoints * sizeof(float3);

	if (3 * size4 + size3 > MAX_GLOB_MEM){
		cout << "Allocated to much Global Memory" << endl; return; }

	cout << "\n	allocating shared memory: " << sharedMemSize << " MAX = 49152 " << endl;
	cout << "	allocating global memory: " << 3 * size4 + size3 << " MAX = 2095251456" <<endl;
	cout << "	allocating grid sizes: " << numTiles << " MAX = 65535" << endl;
	float4* host_points;					// array of float4 point data {m, x, y, z}
	float4* host_velocities;				// initial velocities to be loaded into dev memory
	float3* host_accels;					// accels just used in the GPU

			///*** Device Memory Objects ***///
	
	float4* dev_points;
	float4* dev_velocities;		//examples use velocity float4s 
	float3* dev_accels;			

			///*** Load point data from file into _Pinned_ host memory ***///

	std::string line;
	int i = 0; 
	checkCuda( cudaMallocHost((void**)&host_points, size4) ); 		//pinned
	checkCuda( cudaMallocHost((void**)&host_velocities, size4) ); 	//pinned
	checkCuda( cudaMallocHost((void**)&host_accels, size3) );		//pinned
	while(getline(datafile, line)){
		std::stringstream linestream(line);
		if(i==0)
			std::cout<<std::endl;		 
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
	datafile.close();

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

	SaveStateB(host_points, numPoints, savefile); //initial save
	int nstep;
	cout << "	calling leapstep with: " << numTiles << " gridsize (numblocks), " << threadsPerBlock << " blocksize, and " << sharedMemSize << " shared memory per block" << endl;
	for (nstep = 0; nstep < numSteps; nstep++){

		cout << "	Steps:" << nstep+1 << "\r";
		cout.flush();
		if (nstep % coolStep == 0 && nstep!=0){
			cout << "			COOLING" << endl;
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

	fclose(savefile);
}

=======
>>>>>>> e3977a9aa8cfaf316015eb971540a3df30b629ec
