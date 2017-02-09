/****************************************************************************
 * FILE: mpi_heat2D.c
 * DESCRIPTIONS:  
 *   HEAT2D Example - Parallelized C Version
 *   This example is based on a simplified two-dimensional heat 
 *   equation domain decomposition.  The initial temperature is computed to be 
 *   high in the middle of the domain and zero at the boundaries.  The 
 *   boundaries are held at zero throughout the simulation.  During the 
 *   time-stepping, an array containing two domains is used; these domains 
 *   alternate between old data and new data.
 *
 *   In this parallelized version, the grid is decomposed by the master
 *   process and then distributed by rows to the worker processes.  At each 
 *   time step, worker processes must exchange border data with neighbors, 
 *   because a grid point's current temperature depends upon it's previous
 *   time step value plus the values of the neighboring grid points.  Upon
 *   completion of all time steps, the worker processes return their results
 *   to the master process.
 *
 *   Two data files are produced: an initial data set and a final data set.
 * AUTHOR: Blaise Barney - adapted from D. Turner's serial C version. Converted
 *   to MPI: George L. Gusciora (1/95)
 * LAST REVISED: 04/02/05
 ********************************************************************/
#include "mpi.h"
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "settings.h"

int test_end(int startx, int endx, int starty, int endy, int ny, float *u1, float *u2);

struct Parms { 
  float cx;
  float cy;
} parms = {0.1, 0.1};

int main (int argc, char *argv[])
{
void inidat(), prtdat(), update();
float  u[2][NXPROB][NYPROB];        /* array for grid */
int	taskid,                     /* this task's unique id */
	blockdim,                 /* number of worker processes */
	numtasks,                   /* number of tasks */
	averow,rows,avecol,columns,offsetx,offsety,extrax,extray,   /* for sending rows of data */
	dest, source,               /* to - from for message send-receive */
	left,right,up,down,        /* neighbor tasks */
	msgtype,                    /* for message types */
	rc,startx,endx,starty,endy,               /* misc */
	startx_t,endx_t,starty_t,endy_t, midx, midy,
	i,j,k,ix,iy,iz,it;              /* loop variables */
	MPI_Status status;
	MPI_Comm mpi_comm_cart;
	MPI_Datatype block_type;
	MPI_Datatype column_type;
	int dims[2];
	int periods[2];
	int coords[2];
	int target_coords[2];
	int backup[4];
	double init_time;
	double finalize_time;
	double start_time;
	double end_time;
	int endres; int endval;

	/* First, find out my taskid and how many tasks are running */
	MPI_Init(&argc,&argv);
	init_time = MPI_Wtime();


	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	
	/*find processes per dimesion and create cartesian*/

	dims[0] = 0;
	dims[1] = 0;

	MPI_Dims_create(numtasks, 2, dims);

	periods[0] = 0;
	periods[0] = 0;

	MPI_Cart_create (MPI_COMM_WORLD, 2, dims, periods, 1, &mpi_comm_cart);

	/*find position in cartesian and get neighbors*/

	MPI_Comm_rank(mpi_comm_cart,&taskid);
	MPI_Cart_coords(mpi_comm_cart, taskid, 2, coords);

	coords[0] -= 1;
	if (coords[0] < 0)
		up = MPI_PROC_NULL;
	else
		MPI_Cart_rank(mpi_comm_cart, coords, &up);
	coords[0] += 2;
	if (coords[0] >= dims[0])
		down = MPI_PROC_NULL;
	else
		MPI_Cart_rank(mpi_comm_cart, coords, &down);
	coords[0] -= 1; 
	coords[1] -= 1;
	if (coords[1] < 0)
		left = MPI_PROC_NULL;
	else
		MPI_Cart_rank(mpi_comm_cart, coords, &left);
	coords[1] += 2;
	if (coords[1] >= dims[1])
		right = MPI_PROC_NULL;
	else
		MPI_Cart_rank(mpi_comm_cart, coords, &right);
	coords[1] -= 1;

	/*non-master threads get their block info and data through a vector datatype
	the vector consists of non-continuous arrays of floats*/
	if (taskid != MASTER) {
		for (iz=0; iz<2; iz++)
			for (ix=0; ix<NXPROB; ix++) 
				for (iy=0; iy<NYPROB; iy++) 
					u[iz][ix][iy] = 0.0;

		source = MASTER;
		msgtype = BEGIN;
		MPI_Recv(&offsetx, 1, MPI_INT, source, msgtype, mpi_comm_cart, MPI_STATUS_IGNORE);
		MPI_Recv(&offsety, 1, MPI_INT, source, msgtype, mpi_comm_cart, MPI_STATUS_IGNORE);
		MPI_Recv(&rows, 1, MPI_INT, source, msgtype, mpi_comm_cart, MPI_STATUS_IGNORE);
		MPI_Recv(&columns, 1, MPI_INT, source, msgtype, mpi_comm_cart, MPI_STATUS_IGNORE);

		MPI_Type_vector (rows,columns,NYPROB,MPI_FLOAT,&block_type);
		MPI_Type_commit (&block_type);

		MPI_Recv(&u[0][offsetx][offsety], 1, block_type, source, msgtype, mpi_comm_cart, &status);

		/*compute boundaries of elements that will be processed-because offset can include global table borders*/
		if (offsetx ==0 ) 
			startx = 1;
		else 
			startx = offsetx;
		if ((offsetx+rows)==NXPROB) 
			endx=offsetx+rows-2;
		else 
			endx =offsetx+rows-1;

		if (offsety==0) 
			starty=1;
		else 
			starty=offsety;
		if ((offsety+columns)==NYPROB) 
			endy=offsety+columns-2;
		else 
			endy=offsety+columns-1;
	}

	/*master computes the borders of each block and sends the data*/
	if (taskid == MASTER) {		
		if ((numtasks > MAXWORKER) || (numtasks < MINWORKER)) {
			printf("ERROR: the number of tasks must be between %d and %d.\n",MINWORKER+1,MAXWORKER+1);
			printf("Quitting...\n");
			MPI_Abort(MPI_COMM_WORLD, rc);
			exit(1);
		}
		
		for (ix=0; ix<NXPROB; ix++) 
			for (iy=0; iy<NYPROB; iy++) 
				u[1][ix][iy] = 0.0;

		inidat(NXPROB, NYPROB, u);
		prtdat(NXPROB, NYPROB, u, "initial.dat");
		
		/*average row size dictate minimum number of x-elements, extrax remain*/
		averow = NXPROB/dims[0];
		extrax = NXPROB%dims[0];
		/*average column size dictate minimum number of y-elements*/
		avecol = NYPROB/dims[1];
		extray = NYPROB%dims[1];

		offsetx = 0;

		for (i=1; i<=dims[0]; i++) {
			/*handles cases so that no process gets only part of the border of the table-like the arr[0][y] column*/
			if (extrax > 1 || (extrax > 0 && i == dims[0])) {
				rows = averow+1;
				extrax--;
			} else
				rows = averow;

			offsety = 0;
			extray = NYPROB%dims[1];

			for (j=1; j<=dims[1]; j++) {
				/*same for y*/
				if (extray > 1 || (extray > 0 && j == dims[1])) {
					columns = avecol+1;
					extray--;
				} else
					columns = avecol;
				/*can send data-it is not self referential-else we just save the data for later*/
				if (i-1 != coords[0] || j-1 != coords[1]) {
					target_coords[0] = i-1;
					target_coords[1] = j-1;

					MPI_Cart_rank(mpi_comm_cart, target_coords, &dest);

					MPI_Send(&offsetx, 1, MPI_INT, dest, BEGIN, mpi_comm_cart);
					MPI_Send(&offsety, 1, MPI_INT, dest, BEGIN, mpi_comm_cart);
					MPI_Send(&rows, 1, MPI_INT, dest, BEGIN, mpi_comm_cart);
					MPI_Send(&columns, 1, MPI_INT, dest, BEGIN, mpi_comm_cart);

					MPI_Datatype block_type;
					MPI_Type_vector (rows,columns,NYPROB,MPI_FLOAT,&block_type);
					MPI_Type_commit(&block_type);

					MPI_Send(&u[0][offsetx][offsety], 1, block_type, dest, BEGIN, mpi_comm_cart);
				} else {
					backup[0] = offsetx;
					backup[1] = offsety;
					backup[2] = rows;
					backup[3] = columns;

					if (offsetx ==0 ) 
						startx = 1;
					else 
						startx = offsetx;

					if ((offsetx+rows)==NXPROB) 
						endx=offsetx+rows-2;
					else 
						endx =offsetx+rows-1;

					if (offsety==0) 
						starty=1;
					else 
						starty=offsety;

					if ((offsety+columns)==NYPROB) 
						endy=offsety+columns-2;
					else 
						endy=offsety+columns-1;
				}
				offsety = offsety + columns;
			}
			offsetx = offsetx + rows;
		}

		offsetx = backup[0];
		offsety = backup[1];
		rows = backup[2];
		columns = backup[3];
	}

	/*set up column data types and send/recv requests*/
	MPI_Type_vector (rows,1,NYPROB,MPI_FLOAT,&column_type);
	MPI_Type_commit (&column_type);

	MPI_Request recvu[2];
	MPI_Request recvd[2];
	MPI_Request recvl[2];
	MPI_Request recvr[2];

	MPI_Request sendu[2];
	MPI_Request sendd[2];
	MPI_Request sendl[2];
	MPI_Request sendr[2];

	MPI_Send_init(&u[0][offsetx][offsety], columns, MPI_FLOAT, up, DTAG, mpi_comm_cart, &sendu[0]);
	MPI_Recv_init(&u[0][offsetx-1][offsety], columns, MPI_FLOAT, up, UTAG, mpi_comm_cart, &recvu[0]);
	MPI_Send_init(&u[0][offsetx+rows-1][offsety], columns, MPI_FLOAT, down, UTAG, mpi_comm_cart, &sendd[0]);
	MPI_Recv_init(&u[0][offsetx+rows][offsety], columns, MPI_FLOAT, down, DTAG, mpi_comm_cart, &recvd[0]);
	MPI_Send_init(&u[0][offsetx][offsety], 1, column_type, left, RTAG, mpi_comm_cart, &sendl[0]);
	MPI_Recv_init(&u[0][offsetx][offsety-1], 1, column_type, left, LTAG, mpi_comm_cart, &recvl[0]);
	MPI_Send_init(&u[0][offsetx][offsety+columns-1], 1, column_type, right, LTAG, mpi_comm_cart, &sendr[0]);
	MPI_Recv_init(&u[0][offsetx][offsety+columns], 1, column_type, right, RTAG, mpi_comm_cart, &recvr[0]);

	MPI_Send_init(&u[1][offsetx][offsety], columns, MPI_FLOAT, up, DTAG, mpi_comm_cart, &sendu[1]);
	MPI_Recv_init(&u[1][offsetx-1][offsety], columns, MPI_FLOAT, up, UTAG, mpi_comm_cart, &recvu[1]);
	MPI_Send_init(&u[1][offsetx+rows-1][offsety], columns, MPI_FLOAT, down, UTAG, mpi_comm_cart, &sendd[1]);
	MPI_Recv_init(&u[1][offsetx+rows][offsety], columns, MPI_FLOAT, down, DTAG, mpi_comm_cart, &recvd[1]);
	MPI_Send_init(&u[1][offsetx][offsety], 1, column_type, left, RTAG, mpi_comm_cart, &sendl[1]);
	MPI_Recv_init(&u[1][offsetx][offsety-1], 1, column_type, left, LTAG, mpi_comm_cart, &recvl[1]);
	MPI_Send_init(&u[1][offsetx][offsety+columns-1], 1, column_type, right, LTAG, mpi_comm_cart, &sendr[1]);
	MPI_Recv_init(&u[1][offsetx][offsety+columns], 1, column_type, right, RTAG, mpi_comm_cart, &recvr[1]);

	start_time = MPI_Wtime();

	#pragma omp parallel num_threads(4) private(it,startx_t,endx_t,starty_t,endy_t,iz,midx,midy)
	{
		midx = (startx+endx)/2;
		midy = (starty+endy)/2;

		if (omp_get_thread_num() == 0) {
			startx_t = startx+1; starty_t = starty+1;
			endx_t = midx; endy_t = midy;
		} else if (omp_get_thread_num() == 1) {
			startx_t = startx+1; starty_t = midy+1;
			endx_t = midx; endy_t = endy-1;
		} else if (omp_get_thread_num() == 2) {
			startx_t = midx+1; starty_t = starty+1;
			endx_t = endx-1; endy_t = midy;
		} else {
			startx_t = midx+1; starty_t = midy+1;
			endx_t = endx-1; endy_t = endy-1;
		}

		iz = 0;
		for (it = 1; it <= STEPS; it++) {
			/*start communications-one per thread*/
			if (omp_get_thread_num() == 0) {
				MPI_Start(&sendl[iz]);
				MPI_Start(&recvl[iz]);
			} else if (omp_get_thread_num() == 1) {
				MPI_Start(&sendr[iz]);
				MPI_Start(&recvr[iz]);
			} else if (omp_get_thread_num() == 2) {
				MPI_Start(&sendu[iz]);
				MPI_Start(&recvu[iz]);
			} else {
				MPI_Start(&sendd[iz]);
				MPI_Start(&recvd[iz]);
			}			
			/*main update*/
			update(startx_t,endx_t,starty_t,endy_t,NYPROB,&u[iz][0][0],&u[1-iz][0][0]);

			/*wait receives needed for borders*/
			if (omp_get_thread_num() == 0) {
				MPI_Wait(&recvl[iz],MPI_STATUS_IGNORE);
			} else if (omp_get_thread_num() == 1) {
				MPI_Wait(&recvr[iz],MPI_STATUS_IGNORE);
			} else if (omp_get_thread_num() == 2) {
				MPI_Wait(&recvu[iz],MPI_STATUS_IGNORE);
			} else {
				MPI_Wait(&recvd[iz],MPI_STATUS_IGNORE);
			}
			
			#pragma omp barrier
			/*update borders and wait end of sends-receive need to have ended which is why we need barriers*/
			if (omp_get_thread_num() == 0) {
				update(startx+1,endx-1,starty,starty,NYPROB,&u[iz][0][0],&u[1-iz][0][0]);
				MPI_Wait(&sendl[iz],MPI_STATUS_IGNORE);
			} else if (omp_get_thread_num() == 1) {
				update(startx+1,endx-1,endy,endy,NYPROB,&u[iz][0][0],&u[1-iz][0][0]);
				MPI_Wait(&sendr[iz],MPI_STATUS_IGNORE);
			} else if (omp_get_thread_num() == 2) {
				update(startx,startx,starty,endy,NYPROB,&u[iz][0][0],&u[1-iz][0][0]);
				MPI_Wait(&sendu[iz],MPI_STATUS_IGNORE);
			} else {
				update(endx,endx,starty,endy,NYPROB,&u[iz][0][0],&u[1-iz][0][0]);
				MPI_Wait(&sendd[iz],MPI_STATUS_IGNORE);
			}
			
			#pragma omp barrier

			iz = 1 - iz;
		}
	}

	iz = STEPS%1;

	end_time = MPI_Wtime();
	/*send block boundaries and data back to master*/
	if (taskid != MASTER) {
		MPI_Send(&offsetx, 1, MPI_INT, MASTER, DONE, mpi_comm_cart);
		MPI_Send(&offsety, 1, MPI_INT, MASTER, DONE, mpi_comm_cart);
		MPI_Send(&rows, 1, MPI_INT, MASTER, DONE, mpi_comm_cart);
		MPI_Send(&columns, 1, MPI_INT, MASTER, DONE, mpi_comm_cart);
		printf ("Look up here man %d\n", iz);
		MPI_Send(&u[iz][offsetx][offsety], 1, block_type, MASTER, DONE, mpi_comm_cart);
	}
	/*master collects results as blocks (vector), prints them and prints total time*/
	if (taskid == MASTER) {
		for (i=1; i<=dims[0]; i++) {
			for (j=1; j<=dims[1]; j++) {
				if (i-1 == coords[0] && j-1 == coords[1])
					continue;

				target_coords[0] = i-1;
				target_coords[1] = j-1;

				MPI_Cart_rank(mpi_comm_cart, target_coords, &source);

				msgtype = DONE;
				
				MPI_Recv(&offsetx, 1, MPI_INT, source, msgtype, mpi_comm_cart,MPI_STATUS_IGNORE);
				MPI_Recv(&offsety, 1, MPI_INT, source, msgtype, mpi_comm_cart,MPI_STATUS_IGNORE);
				MPI_Recv(&rows, 1, MPI_INT, source, msgtype, mpi_comm_cart, MPI_STATUS_IGNORE);
				MPI_Recv(&columns, 1, MPI_INT, source, msgtype, mpi_comm_cart, MPI_STATUS_IGNORE);

				MPI_Datatype block_type;
				MPI_Type_vector (rows,columns,NYPROB,MPI_FLOAT,&block_type);
				MPI_Type_commit(&block_type);

				MPI_Recv(&u[0][offsetx][offsety], 1, block_type, source, msgtype, mpi_comm_cart, &status);
			}
		}
		
		for (i = startx; i <= endx; i++)
			for (j = starty; j <= endy; j++)
				u[0][i][j] = u[iz][i][j];
		
		prtdat(NXPROB, NYPROB, &u[0][0][0], "final.dat");

		finalize_time = MPI_Wtime();

		printf ("Total time: %f\n", finalize_time-init_time);
	}

	MPI_Finalize();
}


/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int startx, int endx, int starty, int endy, int ny, float *u1, float *u2) {
	int ix, iy;
	for (ix = startx; ix <= endx; ix++)
		for (iy = starty; iy <= endy; iy++) {
			*(u2+ix*ny+iy) = *(u1+ix*ny+iy) + parms.cx * (*(u1+(ix+1)*ny+iy) + *(u1+(ix-1)*ny+iy) - 2.0 * *(u1+ix*ny+iy)) + parms.cy * (*(u1+ix*ny+iy+1) + *(u1+ix*ny+iy-1) - 2.0 * *(u1+ix*ny+iy));
		}
}


/**************************************************************************
 *  subroutine test_end
 ****************************************************************************/
int test_end(int startx, int endx, int starty, int endy, int ny, float *u1, float *u2) {
	//printf ("Update rectangle %d-%d %d-%d\n", startx, endx, starty, endy);
	int ix, iy;
	int val = 1;
	int local_val;
	for (ix = startx; ix <= endx; ix++) {
		local_val = 1;
		for (iy = starty; iy <= endy; iy++) {
			local_val = local_val & fabs(*(u2+ix*ny+iy) - *(u1+ix*ny+iy)) < EPS;
		}
		val = val & local_val;
	}
	return val;
}


/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, float *u) {
int ix, iy;

for (ix = 0; ix <= nx-1; ix++) 
  for (iy = 0; iy <= ny-1; iy++)
     *(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float *u1, char *fnam) {
int ix, iy;
FILE *fp;

fp = fopen(fnam, "w");
for (iy = ny-1; iy >= 0; iy--) {
  for (ix = 0; ix <= nx-1; ix++) {
    fprintf(fp, "%6.1f", *(u1+ix*ny+iy));
    if (ix != nx-1) 
      fprintf(fp, " ");
    else
      fprintf(fp, "\n");
    }
  }
fclose(fp);
}
