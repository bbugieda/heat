#include "util.h"

/* 
 * XGRID : x dimension of problem grid
 * YGRID : y dimension of problem grid
 * XBLOCK : x dimension of a block (number of threads per block) 
 * YBLOCK : y dimension of a block
 * STEPS : number of times grid is updated
 * PERIOD : (optional) interval of steps in checking for convergence
 * EPSILON : used for float equality
 */


#ifndef XGRID
#define XGRID 256
#endif
#ifndef YGRID
#define YGRID 256
#endif
#ifndef XBLOCK
#define XBLOCK 16
#endif
#ifndef YBLOCK
#define YBLOCK 16
#endif
#ifndef STEPS
#define STEPS 10000
#endif

// Uncomment the following line to enable convergence checks.
// #define PERIOD 20

#ifndef EPSILON
#define EPSILON 0.001f
#endif


void prtdat(const float *u, const char *fnam);
void inidat(float *u);
__global__ void update(const float *old_T, float *new_T);
__global__ void reduce(const float *old_T, const float *new_T,unsigned int *result);
__device__ struct Parms { 
        float cx;
        float cy;
} parms = {0.1f, 0.1f};




int main (void) {
        float *old_T, *new_T, *temp;
        size_t grid_size = sizeof(float)*XGRID*YGRID;

        cudaEvent_t start,stop;
        CUDA_SAFE_CALL(cudaEventCreate(&start));
        CUDA_SAFE_CALL(cudaEventCreate(&stop));

        CUDA_SAFE_CALL(cudaMalloc((void**) (&old_T), grid_size));
        CUDA_SAFE_CALL(cudaMalloc((void**) (&new_T), grid_size));

        temp = static_cast<float *>(malloc(grid_size));
        
        /* Initialize and print grid */
        printf("Grid size: X = %d  Y = %d,  Steps = %d\n",XGRID,YGRID,STEPS);
        #ifdef PERIOD
                printf("Checking for convergence every %d steps (Epsilon = %f)\n", PERIOD, EPSILON);
        #endif
        printf("Initializing grid and writing initial.dat file...\n");
        inidat(temp);
        prtdat(temp, "initial.dat");       

        // Copy initial grid to the device 
        CUDA_SAFE_CALL(cudaMemcpy(old_T, temp, grid_size, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(new_T, old_T, grid_size, cudaMemcpyDeviceToDevice));

        
        // 2D grid of blocks consisting of 2D grids of threads
        // Size of grid and blocks is fixed - get number of blocks in x and y dimensions 
        // to update the grid using one thread per cell  
        unsigned int blocksNumX = CEILING(XGRID,XBLOCK), blocksNumY = CEILING(YGRID,YBLOCK);
        dim3 blocks(blocksNumX,blocksNumY);
        dim3 threads(XBLOCK,YBLOCK);

        #ifdef PERIOD
        		// Arrays at host and device to store results of reduction for blocks 
                unsigned int *reduced, *reduced_d;
                CUDA_SAFE_CALL(cudaMalloc((void**) (&reduced_d), blocksNumX*blocksNumY*sizeof(unsigned int)));
                reduced = static_cast<unsigned int *>(malloc(blocksNumX*blocksNumY*sizeof(unsigned int)));
        #endif
       
        // Start the gpu timer
        CUDA_SAFE_CALL(cudaEventRecord(start, 0));

        // In every step compute updated grid, set it as the input
        // for next step and, if enabled, check for convergence.
        for (unsigned int i=0; i<STEPS; i++){
                #ifdef DEBUG
                	printf("Step %d...\n",i+1);
                #endif
                update<<<blocks,threads>>>(old_T,new_T);
                #ifdef DEBUG
                    CUDA_SAFE_CALL(cudaPeekAtLastError());
                    CUDA_SAFE_CALL(cudaDeviceSynchronize());
                #endif
                swap(old_T,new_T);  // swap input and output buffers

                #ifdef PERIOD
                		// Check for convergence every PERIOD steps by getting
                		// number of unequal cells in each block.
                		// Convergence and halt of iteration if their sum is zero.
                        if (! ((i+1) % PERIOD)) {
                                reduce<<<blocks,threads>>>(old_T,new_T,reduced_d);
                                CUDA_SAFE_CALL(cudaMemcpy(reduced,reduced_d,blocksNumX*blocksNumY*sizeof(unsigned int),cudaMemcpyDeviceToHost));
                                bool found = false;
                                for (unsigned int k = 0; (k < blocksNumX) && !found; k++)
                                    for (unsigned int l = 0; (l < blocksNumY) && !found; l++)
                                        if (reduced[k + l * blocksNumX]) found = true;
                                if (!found) {
                                        printf("Reached convergence at step %d\n",i+1);
                                        break;
                                }
                        }
                #endif 
                
                
        }
        
        CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
        CUDA_SAFE_CALL(cudaMemcpy(temp, old_T, grid_size, cudaMemcpyDeviceToHost));
      
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        float ms;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
        printf("Completed steps in %f milliseconds...\n",ms);


        /* Write final output */
        printf("Writing final.dat file...\n");
        prtdat(temp, "final.dat");

        CUDA_SAFE_CALL(cudaEventDestroy(start));
        CUDA_SAFE_CALL(cudaEventDestroy(stop));
        cudaFree(old_T);
        cudaFree(new_T);
        #ifdef PERIOD
                cudaFree(reduced_d);
                free(reduced);
        #endif
        free(temp);

        printf("Exiting...\n");

        return EXIT_SUCCESS;
}


/* Initialize grid */
void inidat(float *u) {
        for (unsigned int ix = 0; ix <= XGRID-1; ix++) 
                for (unsigned int iy = 0; iy <= YGRID-1; iy++)
                        u[iy*XGRID+ix] = (float)(ix * (XGRID - ix - 1) * iy * (YGRID - iy - 1));
}


/* Print grid to file fnam */
void prtdat(const float *u, const char *fnam) {
        FILE *fp;

        fp = fopen(fnam, "w");
        for (unsigned int iy = 0; iy <= YGRID-1; iy++) {
                for (unsigned int ix = 0; ix <= XGRID-1; ix++) {
                        fprintf(fp, "%6.1f", u[iy*XGRID+ix]);
                        if (ix != XGRID-1) 
                                fprintf(fp, " ");
                        else
                                fprintf(fp, "\n");
                }
        }
        fclose(fp);
}

/* 
 * Update input grid of temperatures. Map every thread to a cell
 * and update every interior grid cell based on neighbor values.
 * Write output values to new_T. 
 */

__global__ void update(const float *old_T, float *new_T) {
        // Map thread and block indices to global cell position
        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

        // Update only interior cells
        if (!x || x >= XGRID - 1 || !y || y >= YGRID - 1) return;  

        unsigned int offset,left,right,top,bottom;
        
        // Buffer offsets of current and neighbor cells        
        offset = x + y * XGRID;
        left = offset - 1;
        right = offset + 1;
        top = offset - XGRID;
        bottom = offset + XGRID;

        
        // Update temperature writing to output array
        new_T[offset] = old_T[offset] +
                        parms.cx * (old_T[left] + old_T[right] - 2.0f * old_T[offset]) +
                        parms.cy * (old_T[top] + old_T[bottom] - 2.0f * old_T[offset]);


}

/* 
 * Compare matrices on block level. Shared memory holds individual thread results
 * (whether cell not equal to the respective one in the other matrix) 
 * and each shared array is reduced to a single sum of mismatches.
 * Store block sums to array of results.
 */

__global__ void reduce(const float *old_T, const float *new_T, unsigned int* result){
        const unsigned int threads = XBLOCK*YBLOCK;
        __shared__ unsigned int shm[threads];	// array of thread booleans and later sums
        unsigned int shm_offset = threadIdx.x + threadIdx.y * XBLOCK;	// position in shared memory
        
        // Map thread and block indices to the global cell position in grid
        unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned int offset = x + y * XGRID;

        // Thread value is 0 if the pair is approximately equal or thread not in grid.
		// Any other case is a valid inequality and value is 1.

        if (x >= XGRID || y >= YGRID) shm[shm_offset] = 0;
        else shm[shm_offset] = ((abs(old_T[offset]-new_T[offset]) < EPSILON) ? 0 : 1);

        __syncthreads();

        // Reduce the array using sequential addressing
        for (unsigned int i = threads/2 ; i > 0 ; i >>= 1) {
                if (shm_offset < i) shm[shm_offset] += shm[shm_offset+i];
                 __syncthreads();
        }
        
        // Total sum (number of mismatches) on first element
        // First thread writes block result to global memory.     
        if (!shm_offset) result[blockIdx.x+blockIdx.y*gridDim.x] = shm[0];
}
