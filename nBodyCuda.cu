#include "nBodyCuda.h" 

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
