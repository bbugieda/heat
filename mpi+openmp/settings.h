#define NXPROB     	256                 	/* x dimension of problem grid */
#define NYPROB      	256              	/* y dimension of problem grid */
#define STEPS       	10000			/* number of time steps */
#define MAXWORKER   	50                  	/* maximum number of worker tasks */
#define MINWORKER   	1                 	/* minimum number of worker tasks */
#define BEGIN       	1                  	/* message tag */
#define LTAG        	2                  	/* message tag */
#define RTAG        	3                  	/* message tag */
#define UTAG		6
#define DTAG		7
#define NONE        	0                 	/* indicates no neighbor */
#define DONE        	4                  	/* message tag */
#define MASTER      	0                  	/* taskid of first process */
#define EPS	    	0.001f			/*minimum diverence to not consider that results converged*/
#define PERIOD	    	20			/*convergence test period*/
