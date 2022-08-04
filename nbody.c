#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/*
 * pRNG based on http://www.cs.wm.edu/~va/software/park/park.html
 */
#define MODULUS    2147483647
#define MULTIPLIER 48271
#define DEFAULT    123456789

static long seed = DEFAULT;

double Random(void)
/* ----------------------------------------------------------------
 * Random returns a pseudo-random real number uniformly distributed 
 * between 0.0 and 1.0. 
 * ----------------------------------------------------------------
 */
{
  const long Q = MODULUS / MULTIPLIER;
  const long R = MODULUS % MULTIPLIER;
        long t;

  t = MULTIPLIER * (seed % Q) - R * (seed / Q);
  if (t > 0) 
    seed = t;
  else 
    seed = t + MODULUS;
  return ((double) seed / MODULUS);
}

/*
 * End of the pRNG algorithm
 */

typedef struct {
    double x, y, z;
    double mass;
    } Particle;
typedef struct {
    double xold, yold, zold;
    double fx, fy, fz;
    } ParticleV;

void InitParticles( Particle[], ParticleV[], int);
double ComputeForces( Particle [], Particle [], ParticleV [], int, int);
double ComputeNewPos( Particle [], ParticleV [], int, double);

int main (int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    double time;
    Particle  * particles;   /* Particles */
    ParticleV * pv;          /* Particle velocity */
    int         npart, i, j, rank, n_procs;
    int         cnt;         /* number of times in loop */
    double      sim_t;       /* Simulation time */
    int tmp;
    tmp = fscanf(stdin,"%d\n",&npart);
    tmp = fscanf(stdin,"%d\n",&cnt);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Bcast(&npart, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Allocate memory for particles */
    particles = (Particle *) malloc(sizeof(Particle)*npart);
    pv = (ParticleV *) malloc(sizeof(ParticleV)*npart);

    int count=4;
    int lengths[4] = {1,1,1,1};
    MPI_Datatype oldtypes[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype MPI_Particle;
    MPI_Aint offsets[4] = {0, sizeof(double), sizeof(double)*2, sizeof(double)*3};
    MPI_Type_create_struct(count, lengths, offsets, oldtypes, &MPI_Particle);
    MPI_Type_commit(&MPI_Particle);

    int countV=6;
    int lengthsV[6] = {1,1,1,1,1,1};
    MPI_Datatype oldtypesV[6] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype MPI_ParticleV;
    MPI_Aint offsetsV[6] = {0, sizeof(double), sizeof(double)*2, sizeof(double)*3, sizeof(double)*4, sizeof(double)*5};
    MPI_Type_create_struct(countV, lengthsV, offsetsV, oldtypesV, &MPI_ParticleV);
    MPI_Type_commit(&MPI_ParticleV);

    int part_per_proc;
    if (rank==0)
      part_per_proc = npart / n_procs;
  
    MPI_Bcast(&part_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);

/* Generate the initial values */
    if (rank==0) {
      double starttime, endtime;
      starttime = MPI_Wtime();
      InitParticles( particles, pv, npart);
      endtime   = MPI_Wtime();
      // printf("Tempo Sequencial %f segundos\n",endtime-starttime);
    }

    MPI_Bcast(particles, npart, MPI_Particle, 0, MPI_COMM_WORLD);
    MPI_Bcast(pv, npart, MPI_Particle, 0, MPI_COMM_WORLD);
    
    double starttime, endtime;
    if (rank==0) starttime = MPI_Wtime();

    sim_t = 0.0;
    Particle  * sub_part;
    ParticleV * sub_pv;
    sub_part = (Particle *) malloc(sizeof(Particle)*part_per_proc);
    sub_pv = (ParticleV *) malloc(sizeof(ParticleV)*part_per_proc);
    
    MPI_Scatter(particles, part_per_proc, MPI_Particle, sub_part, part_per_proc, MPI_Particle, 0, MPI_COMM_WORLD);
    MPI_Scatter(pv, part_per_proc, MPI_ParticleV, sub_pv, part_per_proc, MPI_ParticleV, 0, MPI_COMM_WORLD);
    while (cnt--) {
      double max_f, sub_max_f;
      /* Compute forces (2D only) */
      sub_max_f = ComputeForces( sub_part, particles, sub_pv, npart, n_procs);
      MPI_Allreduce(&sub_max_f, &max_f, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      /* Once we have the forces, we compute the changes in position */
      sim_t += ComputeNewPos( sub_part, sub_pv, part_per_proc, max_f);

      MPI_Allgather(sub_part, part_per_proc, MPI_Particle, particles, part_per_proc, MPI_Particle, MPI_COMM_WORLD);
      MPI_Allgather(sub_pv, part_per_proc, MPI_ParticleV, pv, part_per_proc, MPI_ParticleV, MPI_COMM_WORLD);
    }

    if (rank==0) {
      for (i=0; i<npart; i++)
        fprintf(stdout,"%.5lf %.5lf %.5lf\n", particles[i].x, particles[i].y, particles[i].z);
    }
    
    if (rank==0) {
      endtime   = MPI_Wtime();
      printf("Tempo paralelo %f segundos\n",endtime-starttime);
    }

    free(sub_part);
    free(sub_pv);
    MPI_Type_free(&MPI_Particle);
    MPI_Type_free(&MPI_ParticleV);
    MPI_Finalize();
    return 0;
}

void InitParticles( Particle particles[], ParticleV pv[], int npart )
{
    int i;
    for (i=0; i<npart; i++) {
	particles[i].x	  = Random();
	particles[i].y	  = Random();
	particles[i].z	  = Random();
	particles[i].mass = 1.0;
	pv[i].xold	  = particles[i].x;
	pv[i].yold	  = particles[i].y;
	pv[i].zold	  = particles[i].z;
	pv[i].fx	  = 0;
	pv[i].fy	  = 0;
	pv[i].fz	  = 0;
    }
}

double ComputeForces( Particle myparticles[], Particle others[], ParticleV pv[], int npart, int n_procs)
{
  double max_f;
  int i;
  max_f = 0.0;
  for (i=0; i<npart/n_procs; i++) {
    int j;
    double xi, yi, mi, rx, ry, mj, r, fx, fy, rmin;
    rmin = 100.0;
    xi   = myparticles[i].x;
    yi   = myparticles[i].y;
    fx   = 0.0;
    fy   = 0.0;
    for (j=0; j<npart; j++) {
      rx = xi - others[j].x;
      ry = yi - others[j].y;
      mj = others[j].mass;
      r  = rx * rx + ry * ry;
      /* ignore overlap and same particle */
      if (r == 0.0) continue;
      if (r < rmin) rmin = r;
      r  = r * sqrt(r);
      fx -= mj * rx / r;
      fy -= mj * ry / r;
    }
    pv[i].fx += fx;
    pv[i].fy += fy;
    fx = sqrt(fx*fx + fy*fy)/rmin;
    if (fx > max_f) max_f = fx;
  }
  return max_f;
}

double ComputeNewPos( Particle particles[], ParticleV pv[], int npart, double max_f)
{
  int i;
  double a0, a1, a2;
  static double dt_old = 0.001, dt = 0.001;
  double dt_new;
  a0	 = 2.0 / (dt * (dt + dt_old));
  a2	 = 2.0 / (dt_old * (dt + dt_old));
  a1	 = -(a0 + a2);
  for (i=0; i<npart; i++) {
    double xi, yi;
    xi	           = particles[i].x;
    yi	           = particles[i].y;
    particles[i].x = (pv[i].fx - a1 * xi - a2 * pv[i].xold) / a0;
    particles[i].y = (pv[i].fy - a1 * yi - a2 * pv[i].yold) / a0;
    pv[i].xold     = xi;
    pv[i].yold     = yi;
    pv[i].fx       = 0;
    pv[i].fy       = 0;
  }
  dt_new = 1.0/sqrt(max_f);
  /* Set a minimum: */
  if (dt_new < 1.0e-6) dt_new = 1.0e-6;
  /* Modify time step */
  if (dt_new < dt) {
    dt_old = dt;
    dt     = dt_new;
  }
  else if (dt_new > 4.0 * dt) {
    dt_old = dt;
    dt    *= 2.0;
  }
  return dt_old;
}