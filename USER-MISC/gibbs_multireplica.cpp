/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing Author: Edwin Antillon:  edwin.antillon@gmail.com
   VERSION:    Jul 7  2020
------------------------------------------------------------------------- */


#include <cmath>
#include <cstdlib>
#include <cstring>
#include "gibbs_multireplica.h"
#include "universe.h"
#include "domain.h"
#include "atom.h"
#include "update.h"
#include "integrate.h"
#include "modify.h"
#include "comm_brick.h"
#include "comm_tiled.h"
#include "neighbor.h"
#include "compute.h"
#include "pair.h"
#include "force.h"
#include "output.h"
#include "group.h"
#include "thermo.h"
#include "fix.h"
#include "min.h"
#include "math_const.h"
#include "math_extra.h"
#include "fix_store.h"
#include "random_park.h"
#include "random_mars.h"
#include "finish.h"
#include "timer.h"
#include "memory.h"
#include "error.h"


#include "atom_vec.h"
#include "fix.h"
#include "neighbor.h"



#define PI 3.1415926535

#define MAXENERGYTEST 1.0e50


enum{NONE,MC};
enum{XYZ,XY,YZ,XZ,ANISO,TRICLINIC};
// same as domain.cpp, fix_nvt_sllod.cpp, compute_temp_deform.cpp
enum{NO_REMAP,X_REMAP};

using namespace LAMMPS_NS;
using namespace MathConst;

//#define DEBUG123
//#define DEBUG1 1
//#define DEBUG2 
//#define DEBUG8
//#define DEBUG9 


// wrapper for fortran routine
extern "C" {
  void dgelsd_( int* m, int* n, int* nrhs, double* a, int* lda,
		      double* b, int* ldb, double* s, double* rcond, int* rank,
		      double* work, int* lwork, int* iwork, int* info );
  //  void print_matrix( char* desc, int m, int n, double* a, int lda );

}


/* ---------------------------------------------------------------------- */

GibbsMultiReplica::GibbsMultiReplica(LAMMPS *lmp) : Pointers(lmp) {}

/* ---------------------------------------------------------------------- */

GibbsMultiReplica::~GibbsMultiReplica()
{
 MPI_Comm_free(&roots);

  if (random_equal) delete random_equal;
  delete ranboltz;
  delete [] set_temp;   // 
  delete [] world2root;
  delete [] world_chemistry;
  delete [] molar_fraction;
  delete [] emol;
  delete [] mu;
}

/* ----------------------------------------------------------------------
   perform Gibbs-Ensemble Monte-Carlo moves between multireplicas
------------------------------------------------------------------------- */

void GibbsMultiReplica::command(int narg, char **arg)
{

  me_universe = universe->me;
  MPI_Comm_rank(world,&me);
  nworlds = universe->nworlds;
  nprocs = universe->nprocs;
  iworld = universe->iworld;
  boltz = force->boltz;
  ntypes = atom->ntypes;
  if(me_universe==0){

    printf("narg=%d : %d ",narg);
    for(int r=0; r<narg; r++)
      printf(": arg[%d] = %s ",r,arg[r]);
    printf(" \n ");    
    printf("ntypes =%d and nworlds = %d \n",ntypes,nworlds);

}



  if (universe->nworlds == 1)
    error->all(FLERR,"Must have more than one processor partition to temper");
  if (domain->box_exist == 0)
    error->all(FLERR,"GibbsMultiReplica command before simulation box is defined");

  int nsteps = force->inumeric(FLERR,arg[0]);
  nevery = force->inumeric(FLERR,arg[1]);
  ncycles = force->inumeric(FLERR,arg[2]);
  double temp = force->numeric(FLERR,arg[3]);

  for (whichfix = 0; whichfix < modify->nfix; whichfix++)
    if (strcmp(arg[4],modify->fix[whichfix]->id) == 0) break;
  if (whichfix == modify->nfix)
    error->universe_all(FLERR,"GibbsMultiReplicaing fix ID is not defined");
  //  else
  //    modify->fix[whichfix]->force_reneighbor=1;

  // the following fix style are supported
  if ((strcmp(modify->fix[whichfix]->style,"nve") != 0) &&
      (strcmp(modify->fix[whichfix]->style,"nvt") != 0) &&
      (strcmp(modify->fix[whichfix]->style,"npt") != 0) &&
      (strcmp(modify->fix[whichfix]->style,"gcmc") != 0) &&
      (strcmp(modify->fix[whichfix]->style,"mc/box") != 0))
    error->universe_all(FLERR,"gibbs/multireplica fix is not supported");
  
  lmp->init();

  // hard-code minimization for now
  /*
  if (update->minimize->searchflag)
    error->all(FLERR,"NEB requires damped dynamics minimizer")
  */

  int n = strlen("cg") + 1;
  min_style = new char[n];
  strcpy(min_style,"cg");

  int narg2 = 1;
  char **args = new char*[narg2];
  args[0] = min_style;
  update->create_minimize(narg2,args);
  delete [] args;

  update->etol = 0.00001;
  update->ftol = 0.00001;
  update->max_eval = 1000;
  update->minimize->init();

  seed_swap = force->inumeric(FLERR,arg[5]);
  seed_boltz = force->inumeric(FLERR,arg[6]);


  //
  memory->create(type_list,ntypes,"gibbs/multireplica:type_list");
  if (strcmp(arg[7],"types") == 0) {
    if ((7+3) > narg) error->all(FLERR,"Illegal gibbs/multireplica command");
    type_list[0] = force->numeric(FLERR,arg[8]);
    type_list[1] = force->numeric(FLERR,arg[9]);
    typesflag=1;
  }
  if(type_list[0] == type_list[1])
    typesflag=0;

  slopeflag=500;
  
  //
  if (strcmp(arg[10],"fudge") == 0) {
    fudge = force->numeric(FLERR,arg[11]);
  }


  //flip_nmax=5;
  flip_nmax=ceil(pow(2*atom->natoms,0.333333));
  Ntot=0;

  // create FixStore object to store revert state                                                                               
  // only used during exchange step. 
  narg2 = 6;
  args = new char*[narg2];
  args[0] = (char *) "gibbs_revert";
  args[1] = (char *) "all";
  args[2] = (char *) "STORE";
  args[3] = (char *) "peratom";
  args[4] = (char *) "0";
  args[5] = (char *) "7";
  modify->add_fix(narg2,args);
  fix_revert = (FixStore *) modify->fix[modify->nfix-1];
  delete [] args;
  
  
  set = new Set[6];
  for (int i = 0; i < 6; i++) {
    set[i].p_start = set[i].p_stop = 0.0;
    set[i].style = NONE;
    set[i].amplitude = 0.0;
  }

  // hardcode ISO Box/MC for now
  double ptarget=0.0;
  double vmove=0.20;

  remapflag = X_REMAP;

  press_couple = XYZ;
  set[0].p_start = set[1].p_start = set[2].p_start = ptarget;
  set[0].p_stop = set[1].p_stop = set[2].p_stop = ptarget;
  set[0].amplitude = set[1].amplitude = set[2].amplitude = vmove;
  set[0].style = set[1].style = set[2].style = MC;
  if (dimension == 2) {
    set[2].p_start = set[2].p_stop = set[2].amplitude = 0.0;
    set[2].style = NONE;
  }


  // swap frequency must evenly divide total # of timesteps
  if (nevery <= 0)
    error->universe_all(FLERR,"Invalid frequency in temper command");
  nmoves = nsteps/nevery;
  if (nmoves*nevery != nsteps)
    error->universe_all(FLERR,"Non integer # of swaps in temper command");


  update->nsteps = nsteps;
  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + nsteps;
  if (update->laststep < 0)
    error->all(FLERR,"Too many timesteps");


  int igroupall = group->find("all");
  groupbitall |= group->bitmask[igroupall];



  // ptrs to compute from thermo
  // notify compute it will be called at first swap

  int id_pe = modify->find_compute("thermo_pe");
  if (id_pe < 0) error->all(FLERR,"gibbs/multireplica could not find thermo_pe compute");
  c_pe = modify->compute[id_pe];
  c_pe->addstep(update->ntimestep + nevery);

  int id_press = modify->find_compute("thermo_press");
  if (id_press < 0) error->all(FLERR,"gibbs/multireplica could not find thermo_press compute");
  c_press = modify->compute[id_press];
  c_press->addstep(update->ntimestep + nevery);


  int id_temp = modify->find_compute("thermo_temp");
  if (id_temp < 0) error->all(FLERR,"gibbs/multireplica could not find thermo_temp compute");
  c_temp = modify->compute[id_temp];
  c_temp->addstep(update->ntimestep + nevery);


  // Create comunication for each world
  int color = me;
  if (me == 0) color = 0;
  else color = 1;
  MPI_Comm_split(universe->uworld,color,0,&roots);


  //  unique RNG for universe, same for all procs
  if (seed_swap) 
    random_equal = new RanPark(lmp,seed_swap);
  else 
    error->universe_all(FLERR,"invalid seed_swap number, needs to be non-zero");
  
  //  unique RNG for this processor
  ranboltz = new RanPark(lmp,seed_boltz + me_universe);
  for (int i = 0; i < 100; i++) ranboltz->uniform();

  //  unique RNG for each world 
  ranrepl  = new RanMars(lmp,seed_swap+iworld);

  world2root = new int[nworlds];
  root2world = new int[nworlds];
  if (me == 0) {
    MPI_Allgather(&me_universe,1,MPI_INT,world2root,1,MPI_INT,roots);
    for (int i = 0; i < nworlds; i++)  root2world[world2root[i]] = i;
  }
  MPI_Bcast(root2world,nworlds,MPI_INT,0,world);
  MPI_Bcast(world2root,nworlds,MPI_INT,0,world);


  molar_fraction = new double[nworlds];
  wmolfrac = new double[nworlds];
  emol = new double[nworlds];

  init_conc = new double[ntypes];
  mu_vect = new double[ntypes];
  nchem = new double[ntypes];
  nchem_local = new double[ntypes];
  world_chemistry = new double[ntypes];

  mu = new double[nworlds];
  farray = new double[3];  
  barray = new double[3];  
  carray = new double[3];  
  darray = new double[3];  
  marray = new double[2];  


  // According to Gibbs Phase Rule
  // world arrays = type arrays
  // Gibbs Phase rule  F=C-P+2 
  // F =2:  Temp, and Press fixed 
  // Hence C=P                                                                                                                                       

  // initialize world arrays
  for(int k=0;k<nworlds;k++){
    mu[k]=0.0;
    molar_fraction[k]=(double)1.0/nworlds;
    wmolfrac[k]=0.0;
  }

  for(int k=0;k<ntypes;k++){
    nchem[k]=0.0;
    nchem_local[k]=0.0;
    world_chemistry[k]=0.0;
  }



  
  // initialize global variables
  molene=0.0;
  wtmp=-1.0;
  werr=0.0;

  farray[0]=0.0;
  farray[1]=0.0;
  farray[2]=0.0;

  barray[0]=0.0;
  barray[1]=0.0;
  barray[2]=0.0;


  carray[0]=0.0;
  carray[1]=0.0;
  carray[2]=0.0;

  darray[0]=0.0;
  darray[1]=0.0;
  darray[2]=0.0;

  marray[1]=0.0;
  marray[2]=0.0;


#ifdef DEBUG0
  if (me_universe == 0){
    printf("world 2 root : ");
    for(int k=0;k<nworlds;k++)
      printf(" %d ",world2root[k]);
    printf("\n");
  }
#endif


  // set constant volume for reference
  vol0 = domain->xprd * domain->yprd * domain->zprd;
  

  conc_matrix = new double[nworlds*ntypes];
  mu_matrix = new double[nworlds*ntypes];
  for(int i=0;i<nworlds;i++)
    for(int j=0;j<ntypes;j++){
      conc_matrix[i*nworlds+j]=0.0;
      mu_matrix[i*nworlds+j]=0.0;
    }
  mupair = new double[ntypes*ntypes];
  muavg  = new double[ntypes*ntypes];


  fatt = new double[ntypes*ntypes];
  facc = new double[ntypes*ntypes];

  for (int i = 0; i < ntypes; i++)
    for (int j = 0; j < ntypes; j++){
      mupair[i*ntypes+j]=0; 
      muavg[i*ntypes+j]=0.0; 
      facc[i*ntypes+j]=0.0; 
      fatt[i*ntypes+j]=0.0; 
    }
    


  // running average on dmu
  runavg = new double*[5];
  for (int i=0;i<5;i++){
    runavg[i] = new double[ntypes*ntypes];  
    for (int j = 0; j < (ntypes*ntypes); j++)
      runavg[i][j] = 0.0;
  }



  // create static list of set temperatures
  // allgather tempering arg "temp" across root procs
  // bcast from each root to other procs in world
  set_temp = new double[nworlds];
  if (me == 0) MPI_Allgather(&temp,1,MPI_DOUBLE,set_temp,1,MPI_DOUBLE,roots);
  MPI_Bcast(set_temp,nworlds,MPI_DOUBLE,0,world);


#ifdef DEBUG0
  if (me_universe == 0){
    printf("temps : ");
    for(int k=0;k<nworlds;k++)
      printf(" %f ",set_temp[k]);
    printf("\n");
  }
#endif



#ifdef DEBUG0
  if (me_universe == 0){
    printf("temp 2 world : ");
    for(int k=0;k<nworlds;k++)
      printf(" %d ",temp2world[k]);
    printf("\n");
  }
#endif

  time_comm =0.0;
  atom_all_nmax = 0;
  atom_i_intra_nmax = 0;
  atom_j_intra_nmax = 0;
  atom_many_nmax = 0;
  local_all_atoms_list = NULL;
  local_swap_iatom_list = NULL;
  local_swap_jatom_list = NULL;
  local_katoms_list = NULL;

  // read options from end of input line            
  // setup tempering runs

  int which, partner;
  //  int swap_inter,swap_intra,flip_inter,vol_inter,exch_inter;
  int world1=-1,world2=-1;
  int swap_type[2],world_pair[2];
  double ranmove,werr_partner,wemol;

  if (me_universe == 0 && universe->uscreen)
    fprintf(universe->uscreen,"Setting up tempering ...\n");
  
  if (me_universe == 0) {
    if (universe->uscreen) {
      fprintf(universe->uscreen,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->uscreen," W%d",i);
      fprintf(universe->uscreen,"\n");
    }
    if (universe->ulogfile) {
      fprintf(universe->ulogfile,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->ulogfile," W%d",i);
      fprintf(universe->ulogfile,"\n");
    }
    print_status();
  }

  timer->init();
  timer->barrier_start(); 


  int widom_flag=0;
  double vol_acc_rate=0.0;
  double flip_acc_rate=0.0;
  double intra_acc_rate=0.0;

  dmu=0.0;
  // initiallize counters 


  sinter_att=1;     sintra_att=1;  flip_att=1;   vol_att=1;  exch_att=1;
  sinter_suc=0;     sintra_suc=0;  flip_suc=0;   vol_suc=0;  exch_suc=0;

  for (int iswap = 0; iswap < nmoves; iswap++) {

    swap_inter=0;         swap_intra=0;      flip_inter=0;     vol_inter=0;     exch_inter=0;
    swap_inter_all=0;     swap_intra_all=0;  flip_inter_all=0; vol_inter_all=0; exch_inter_all=0;
    
    world_pair[0]= world_pair[1] =-1;
    swap_type[0] = swap_type[1]  =-1;
    partner = -1;
    which=iswap;

    // run for nevery timesteps
    dynamics();  // MD
    //minimization(nevery); // Minimization

    // store energy for each multireplica before attempts
    energy_stored = energy_full(1); 
    if(domain->dimension == 3)
      vol_stored = domain->xprd * domain->yprd * domain->zprd;
    else
      error->all(FLERR,"dimensions in volume are inconsistent");            
    dens_stored =force->mv2d*group->mass(0)/vol_stored;

    
    ////////////////////
    // track molar energy 
    wemol = energy_stored*molar_fraction[iworld];
    if (me == 0)    // gather new info
      MPI_Allgather(&wemol,1,MPI_DOUBLE,emol,1,MPI_DOUBLE,roots);        
    MPI_Bcast(emol,nworlds,MPI_DOUBLE,0,world);
    
    molene=0.0;
    for(int i=0;i<nworlds;i++){
      molene += emol[i]*nworlds; 
    }
    /////////////////////////

    // global root decides which worlds and what species to swap                                                        
    // the type species might or might not be used in this cycle
    if(me_universe==0){
      world_pair[0] = ranboltz->uniform()*nworlds;
      world_pair[1]=world_pair[0];
      while(world_pair[1] == world_pair[0]){
        world_pair[1] = ranboltz->uniform()*nworlds;
      }
      
      // pick types to swap                                                                                                     
      while(swap_type[0]==swap_type[1]){
	swap_type[0]  = (ranboltz->uniform()*ntypes)+1;
	swap_type[1]  = (ranboltz->uniform()*ntypes)+1;
      }
      
      
      // pick random MC-move
      ranmove=ranboltz->uniform();
      if(ranmove<0.3)      
	mcflag =0;   // intra swap
      else if(ranmove<0.6) 
	mcflag =1;   // inter swap
      else if(ranmove<1.0)  // .90
	mcflag =2;   // flip 
      //	else if(ranmove<0.95) 
      else
	mcflag =4;// volume
    }
    
    
    if(mcflag==2 && iswap<500)
      mcflag=1;
    
    // communicate pairs to entire universe from global root
    MPI_Bcast(world_pair,2,MPI_INT,0,universe->uworld);
    MPI_Bcast(swap_type,2,MPI_INT,0,universe->uworld);
    MPI_Bcast(&mcflag,1,MPI_INT,0,universe->uworld);
    
    if(world_pair[0]>-1 && world_pair[1]>-1){
      world1=world2root[world_pair[0]];
      world2=world2root[world_pair[1]];
      if (me_universe==world1){
	partner=world2;
      }else if (me_universe==world2){
	partner=world1;
      }
    }
    
    
    //Broadcast partner from root to world
    MPI_Bcast(&partner,1,MPI_INT,0,world);

    // count chemistry
    count_chemistry(); 
    MPI_Allreduce(nchem,world_chemistry,universe->nworlds,MPI_DOUBLE,MPI_SUM,roots);
    
    // set initial conc. matrix                                                                       
    if(iswap==0){
      for(int k=0;k<nworlds;k++){
        init_conc[k]=(double)world_chemistry[k]/nworlds;
	Ntot+=(double)world_chemistry[k];
	if(me_universe==0)
	  printf("initial concentration[%d] is %f  \n",k,init_conc[k]);
      }
      if(me_universe==0)
	printf("total number of particles is %f  \n",Ntot);
    }


    update_swap_atoms_list(swap_type); // sets niswap njswap
    update_all_atoms_list(); // sets nxchg

    // adjust volume moves to approach target acceptance rate 
    if(iswap>500 && iswap%500==0 && vol_att>0){
      vol_acc_rate = (double)vol_suc/vol_att;

	// reset counters every 5 vol change attempts.
	if(iswap%5000==0){
	  vol_suc=0;
	  vol_att=1;
	}

      if(vol_acc_rate>0.10)
	vmove /= 0.90;
      else
	vmove *=0.90;
      set[0].amplitude = set[1].amplitude = set[2].amplitude = vmove;
    }
    

    
    if(typesflag)    
      if(iswap>1500 && iswap%1000==0 && flip_att>0){
	flip_acc_rate = (double)flip_suc/flip_att;
	intra_acc_rate = (double)sintra_suc/sintra_att;

	// reset counters every 5 flip_max change attempts.
	if(iswap%5000==0){
	  flip_suc=0;
	  flip_att=1;
	}

	if(flip_acc_rate==0.0){
	  flip_nmax=ceil(random_equal->uniform()*pow(2*atom->natoms,0.333333));
	  if(check_molfrac(wmolfrac))
	    for(int i=0;i<nworlds;i++)
	      molar_fraction[i]=wmolfrac[i];
	}else if(flip_acc_rate>0.20)
	  flip_nmax++;
	else
	  flip_nmax--;
	//	else
	//	  flip_nmax--;
	
	if(flip_nmax<1)
	  flip_nmax=1;
	
	if(flip_nmax>atom->natoms/2)
	  flip_nmax=atom->natoms/2;

	//	if(flip_nmax>30)
	//	  
	
      }
    
    if(me_universe==0)
      if(ranboltz->uniform()<0.03)
	widom_flag=1;
      else
	widom_flag=0;
    // communicate flag to entire universe from global root
    MPI_Bcast(&widom_flag,1,MPI_INT,0,universe->uworld);
    

    // estimate chemical potential difference 
    if(typesflag)    
      if(iswap>250 && widom_flag){
	if(me_universe==0)
	  printf("Doing Widom Test \n");
	Widom_test(set_temp[iworld],ncycles,iswap,partner);

	if(me_universe==0){
	  printf("Printing mu matrix \n");
	  for (int i = 0; i < ntypes; i++){
	    for (int j = 0; j < ntypes; j++)
	      printf(" %2.4f ",mupair[i*ntypes+j]);
	    printf("\n");
	  }
	}
	
	for(int i=0;i<(ntypes*ntypes);i++){
	  runavg[4][i]=mupair[i];
	  for(int k=0;k<4;k++)	
	    runavg[k][i]=runavg[k+1][i];
	}
		
	
	if(iswap>1000){
	  slopeflag=1;

	  if (me == 0)    
	    MPI_Allgather(nchem,ntypes,MPI_DOUBLE,conc_matrix,ntypes,MPI_DOUBLE,roots);
	  MPI_Bcast(conc_matrix,ntypes*nworlds,MPI_DOUBLE,0,world);
	  inverse(conc_matrix,molar_fraction,init_conc,0.0);

	  if(me==0){ // Write n(n-1)/2 Chemical potential pairs
	    char s[100] = {0};
	    sprintf(s, "chempot_prc%d.txt", me_universe); 
	    FILE *fp;
	    fp = fopen(s, "a+");
	    fprintf(fp," %d ",iswap);

	    for(int i=0;i<ntypes;i++)
	      for(int j=i+1;j<ntypes;j++){
		fprintf(fp," %2.3f ",mupair[i*ntypes+j]);
	      }

	    fprintf(fp," | ");

	    for(int i=0;i<ntypes;i++)
	      for(int j=i+1;j<ntypes;j++){
		fprintf(fp," %2.3f ",muavg[i*ntypes+j]);
	      }


	    fprintf(fp," | ");

	    for(int i=0;i<nworlds;i++)
	      fprintf(fp," %2.3f ",molar_fraction[i]);
	    
	    if(nworlds==2)  //add analytical soln if binary
	      fprintf(fp," %f %f",(conc_matrix[2]-init_conc[0])/(conc_matrix[2]-conc_matrix[0]),(-conc_matrix[0]+init_conc[0])/(conc_matrix[2]-conc_matrix[0]));
	    
	    fprintf(fp,"\n");
	    fclose(fp);
	  }
	  
	  
	  // compute running average on chem. potential difference
	  for(int i=0;i<(ntypes*ntypes);i++){
	    muavg[i]=0.0;
	    for(int j=0;j<5;j++)
	      muavg[i]+=runavg[j][i]/5.0;
	  }
	}
      }   
    
    

    if(typesflag){   // different types 	
      if(mcflag==0){         //  intra swap 
	swap_intra = attempt_intra_swap(swap_type,set_temp[iworld]);
	sintra_att +=nworlds;
      }else if(mcflag==1){   //  inter swap
	swap_inter = attempt_inter_swap(world_pair,partner,set_temp[iworld],swap_type);
	sinter_att +=nworlds;	  
      }else if(mcflag==2){   //  flip
	flip_inter = attempt_flip(iswap,world_pair,partner,set_temp[iworld],dmu);
	flip_att +=nworlds;
      }else if(mcflag==3){   //  exchange
	exch_inter = attempt_exchange(world_pair,partner,set_temp[iworld]);
	exch_att += nworlds;
      }else{                   //  volume changes
	vol_inter  = attempt_volume_moves(world_pair,partner,set_temp[iworld]);
	vol_att  +=  nworlds;
      }
    }else{ // same type
      if(mcflag==3){  //  exchange
	exch_inter = attempt_exchange(world_pair,partner,set_temp[iworld]);
	exch_att +=nworlds;
      }else{          //  volume changes
	vol_inter = attempt_volume_moves(world_pair,partner,set_temp[iworld]);
	vol_att +=nworlds;
      }
    }


    int which_all = 0;
    MPI_Allreduce(&which,&which_all,1,MPI_INT,MPI_MAX,universe->uworld);

    // Gather success statistics
    MPI_Allreduce(&swap_intra,&swap_intra_all,1,MPI_INT,MPI_SUM,roots);
    MPI_Allreduce(&swap_inter,&swap_inter_all,1,MPI_INT,MPI_SUM,roots);
    MPI_Allreduce(&flip_inter,&flip_inter_all,1,MPI_INT,MPI_SUM,roots);
    MPI_Allreduce(&exch_inter,&exch_inter_all,1,MPI_INT,MPI_SUM,roots);
    MPI_Allreduce(&vol_inter ,&vol_inter_all ,1,MPI_INT,MPI_SUM,roots);

    
    sintra_suc +=swap_intra_all;     
    sinter_suc +=swap_inter_all;
    flip_suc +=flip_inter_all;     
    exch_suc +=exch_inter_all;     
    vol_suc +=vol_inter_all;     
    

    // print out current swap status
    if (me_universe == 0){
      print_status();
      /*
	char s[100] = {0};
	sprintf(s, "file_out.txt"); 
	FILE *fp;
	fp = fopen(s, "a+");
	fprintf(fp, "iswap=%d  world1=%d world2=%d \n",iswap,world1,world2);
	fclose(fp);
      */
    }
    
    
  }

  timer->barrier_stop();
  time_comm += timer->get_wall(Timer::TOTAL);

  update->whichflag = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;

  
  Finish finish(lmp);
  finish.end(1);
  
}



/* ----------------------------------------------------------------------
   proc 0 prints current status
------------------------------------------------------------------------- */

void GibbsMultiReplica::print_status()
{
  
  if(update->ntimestep==0)
    return;
  
  if (universe->uscreen) {
    fprintf(universe->uscreen,BIGINT_FORMAT,update->ntimestep/nevery);
    fprintf(universe->uscreen,"%3d %2.3f %2.3f %2.3f %2.3f %2.3f",
	    mcflag,
	    (double)sintra_suc/sintra_att,	    
	    (double)sinter_suc/sinter_att,
	    (double)flip_suc/flip_att,
	    (double)exch_suc/exch_att,
	    (double)vol_suc/vol_att
	    );
    
    fprintf(universe->uscreen," ");
    for (int i = 0; i < nworlds; i++)
      fprintf(universe->uscreen," %f",world_chemistry[i]);

    fprintf(universe->uscreen," | ");
    for (int i = 0; i < nworlds; i++){
      fprintf(universe->uscreen," %2.3f",molar_fraction[i]);
    }


    fprintf(universe->uscreen," | %2.3f %d ",molene,flip_nmax);
    fprintf(universe->uscreen,"\n");
  }

  if (universe->ulogfile) {
    fprintf(universe->ulogfile,BIGINT_FORMAT,update->ntimestep/nevery);
    fprintf(universe->ulogfile,"%3d %2.3f %2.3f %2.3f %2.3f %2.3f",
	    mcflag,
	    (double)sintra_suc/sintra_att,	    
	    (double)sinter_suc/sinter_att,
	    (double)flip_suc/flip_att,
	    (double)exch_suc/exch_att,
	    (double)vol_suc/vol_att
	    );
    fprintf(universe->ulogfile," ");


    int nchem_sum=0;
    for (int i = 0; i < nworlds; i++){
      fprintf(universe->ulogfile," %f",world_chemistry[i]);
      nchem_sum +=world_chemistry[i];
    }


    fprintf(universe->ulogfile," = %d  | ",nchem_sum);

    for (int i = 0; i < nworlds; i++){
      fprintf(universe->ulogfile," %2.3f",molar_fraction[i]);
    }

    fprintf(universe->ulogfile," | %2.3f %d ",molene,flip_nmax);
    fprintf(universe->ulogfile,"\n");
    fflush(universe->ulogfile);
  }
}

/* ----------------------------------------------------------------------
   dynamics run:
   ------------------------------------------------------------------------- */

void GibbsMultiReplica::dynamics()
{
  /* This piece is borrowed from prd.cpp */ 

  update->whichflag = 1;
  update->nsteps = nevery;

  lmp->init();
  update->integrate->setup(1);

  // this may be needed if don't do full init                                                                                     //modify->addstep_compute_all(update->ntimestep);                                                                                                   
  bigint ncalls = neighbor->ncalls;

  timer->barrier_start();
  update->integrate->run(nevery);
  timer->barrier_stop();
  //  time_category += timer->get_wall(Timer::TOTAL);
  //  nbuild += neighbor->ncalls - ncalls;
  //  ndanger += neighbor->ndanger;

  update->integrate->cleanup();

  //  Is this needed,    Finish for timing output                                                                               
  //  finish->end(0);

}





/* ----------------------------------------------------------------------
   minimization run:
   ------------------------------------------------------------------------- */

void GibbsMultiReplica::minimization()
{
  update->whichflag = 0;  // minimization flag
  update->nsteps = nevery;

  lmp->init();
  update->minimize->setup();

  // this may be needed if don't do full init                                                                                    //modify->addstep_compute_all(update->ntimestep);                                                                                                  
  bigint ncalls = neighbor->ncalls;

  timer->barrier_start();
  update->minimize->run(nevery);
  timer->barrier_stop();

  // does not work!!!
  //  update->minimize->cleanup();

}


/* ----------------------------------------------------------------------
   compute system enthalpy or potential energy 
   ------------------------------------------------------------------------- */

double GibbsMultiReplica::energy_full(int flag)
{

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;  
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);


  
  int eflag = 1;
  int vflag = 0;

  if (modify->n_pre_force) modify->pre_force(vflag);
  if (force->pair) force->pair->compute(eflag,vflag);

  if (force->newton) {
    comm->reverse_comm();
  }


  //  if (force->kspace) force->kspace->compute(eflag,vflag);
  if (modify->n_post_force) modify->post_force(vflag);
  if (modify->n_end_of_step) modify->end_of_step();
  update->eflag_global = update->ntimestep;

  // subtract vol0, to protect precision of p*(vnew-vold)                                                           
  //  total_energy + = p_hydro*(volume-vol0) / force->nktv2p;

  double total_enthalpy = c_pe->compute_scalar();
  /*
  if(flag==0)
    total_enthalpy= c_pe->compute_scalar();
  else
    total_enthalpy= c_pe->compute_scalar() + c_press->compute_scalar()*(vol_stored-vol0)/(force->nktv2p);
  */
  return total_enthalpy;
}



/* 
   attempt coupled volume moves between replicas
*/

int GibbsMultiReplica::attempt_volume_moves(int pair[],int partner,double world_temp){


#ifdef DEBUG1
  printf("VOL_MOVES  me=%d: \n",me_universe);
#endif


  int i;
  double factor;
  double eold = energy_stored;  
  double vold = vol_stored;

  // ignore tricyclic 
  // and rigid bodies
  // for now

 
  for (i = 0; i < 3; i++) {
    if (set[i].style == MC) {
      // same rnd number in all processor
      if(typesflag==0)  // same type , cannot constraint pressure according to Gibbs phase rule. Only Temperature. or viceversa
	factor=(random_equal->uniform()-0.5); 
      else
	factor=(ranrepl->uniform()-0.5); 
      // random walk in cartesian dimensions                                                                      
      set[i].delta = set[i].amplitude*factor;
      if(press_couple == XYZ && i>0)  // for ISO use same variation in all dimensions                             
        set[i].delta = set[i-1].delta;
    }
  }


  for (i = 0; i < 3; i++) {
    if (set[i].style == MC) {
      if(iworld==pair[0]){
	set[i].lo_target = domain->boxlo[i]-set[i].delta;
	set[i].hi_target = domain->boxhi[i]+set[i].delta;
      }else{
	set[i].lo_target = domain->boxlo[i]+set[i].delta;
	set[i].hi_target = domain->boxhi[i]-set[i].delta;
      }
    }
  }


  // convert atoms and rigid bodies to lamda coords                                                                                               
  if (remapflag == X_REMAP) {
    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    for (i = 0; i < nlocal; i++)
      //if (mask[i] & groupbit)
      if (atom->mask[i]) {   // add bitgroup
        domain->x2lamda(x[i],x[i]);
      }
  }

  // reset global and local box to new size/shape   
  // only if mcvol fix is controlling the dimension                                                                                                                         

  if (set[0].style) {
    domain->boxlo[0] = set[0].lo_target;
    domain->boxhi[0] = set[0].hi_target;
  }
  if (set[1].style) {
    domain->boxlo[1] = set[1].lo_target;
    domain->boxhi[1] = set[1].hi_target;
  }
  if (set[2].style) {
    domain->boxlo[2] = set[2].lo_target;
    domain->boxhi[2] = set[2].hi_target;
  }

  domain->set_global_box();
  domain->set_local_box();

  // convert atoms back to box coords                                                                                             
  // ignore ridig for now
 if (remapflag == X_REMAP) {
    double **x = atom->x;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    for (i = 0; i < nlocal; i++)
      if (mask[i])// & groupbit)
        domain->lamda2x(x[i],x[i]);
 }

  // redo KSpace coeffs since box has changed                                                                                     //  
  //  if (kspace_flag) force->kspace->setup();
  // Monte Carlo moves are conditional on energy change                                                                                                                    
 double enew = energy_full(1);  // SEE MC/BOX  COMPUTE ENERGY
 double vnew = domain->xprd * domain->yprd * domain->zprd;

 //  decide acceptance
 //  Prob ~ exp[ - beta( delta E_I  + delta E_II + N_I kT  ln V^I_new/V^I_old + N_II kT  ln V^II_new/V^II_old) ]
 //  A.Z. Panagiotopoulos , N. Quirke , M. Stapleton & D.J. Tildesley
 //  MOLECULAR PHYSICS, 1988, VOL. 63, NO. 4, 527-545
 // nvolume_attempts += 1.0;
 
 double wpe,wpe_partner;
 int metro_success=0;

 wpe = -(enew-eold)/(force->boltz*world_temp) + atom->natoms*log(vnew/vold);

 // hi proc sends PE to low proc                                                                                  
 // lo proc makes the sum and communicates it back to hi proc                                                     
 if(me==0){ // only root processor participate in decision making logic                                           
   if (me_universe > partner)
     MPI_Send(&wpe,1,MPI_DOUBLE,partner,0,universe->uworld);
   else
     MPI_Recv(&wpe_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
   if (me_universe < partner){
     if(ranboltz->uniform() <
        exp(wpe + wpe_partner))
       metro_success=1;
     else
       metro_success=0;
     MPI_Send(&metro_success,1,MPI_INT,partner,0,universe->uworld);
   }else
     MPI_Recv(&metro_success,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
 }


 MPI_Bcast(&metro_success,1,MPI_INT,0,world);

 // synchronize success results via allreduce to the world                                           
 int metro_success_all = 0;
 MPI_Allreduce(&metro_success,&metro_success_all,1,MPI_INT,MPI_MAX,world);

 if (!metro_success) {
   //flip = 0;   (trycilic case ) 
   //next_reneighbor = -1;  

   for (i = 0; i < 3; i++) {
     if (set[i].style == MC) {
      if(iworld==pair[0]){
	set[i].lo_target = domain->boxlo[i]+set[i].delta;
	set[i].hi_target = domain->boxhi[i]-set[i].delta;
      }else{
	set[i].lo_target = domain->boxlo[i]-set[i].delta;
	set[i].hi_target = domain->boxhi[i]+set[i].delta;
      }
     }
   }
 
   // convert atoms and rigid bodies to lamda coords 

   if (remapflag == X_REMAP) {
     double **x = atom->x;
     int *mask = atom->mask;
     int nlocal = atom->nlocal;

     for (i = 0; i < nlocal; i++)
       if (mask[i]) //  & groupbit)
	 domain->x2lamda(x[i],x[i]);
   }

   // reset global and local box to new size/shape                                                                   // only if mcvol fix is controlling the dimension                                                                            
   if (set[0].style) {
     domain->boxlo[0] = set[0].lo_target;
     domain->boxhi[0] = set[0].hi_target;
   }
   if (set[1].style) {
     domain->boxlo[1] = set[1].lo_target;
     domain->boxhi[1] = set[1].hi_target;
   }
   if (set[2].style) {
     domain->boxlo[2] = set[2].lo_target;
     domain->boxhi[2] = set[2].hi_target;
   }


   domain->set_global_box();
   domain->set_local_box();

   // convert atoms and rigid bodies back to box coords                                                                        

   if (remapflag == X_REMAP) {
     double **x = atom->x;
     int *mask = atom->mask;
     int nlocal = atom->nlocal;

     for (i = 0; i < nlocal; i++)
       if (mask[i])// & groupbit)
	 domain->lamda2x(x[i],x[i]);

   }

   // redo KSpace coeffs since box has changed                                                                    
   //   if (kspace_flag) force->kspace->setup();
   // Recompute energy so thermo output and forces are correct.                                                      
   // This could be avoided if all forces and energies were saved.                                                
   energy_full(1);
   return 0;
 }
 
 return 1;
}



/* 
   Widom test 
   compute change in chemical potential. see Frenkel and Smit Eq (9.1.3) 2nd Edition      
*/


void GibbsMultiReplica::Widom_test(double world_temp,int ncycles,int iswap,int partner)
{

 
  // pick random particle to flip 
  int i,j,k = -1,m;
  int ktype=0;  // type of random particle 
  int ftype=-1; // flip to this type
  int num_selected,num_selected_all;
  double delta_energy,ratio,pe;
  double f0=0.0,f1=0.0;
  double gas_mass=0;


  int index=-1;
  const int ttypes = ntypes*ntypes;
  double *loc_cnt = new double[ttypes];
  double *glb_cnt = new double[ttypes];
  double *loc_tmp = new double[ttypes];
  double *glb_tmp = new double[ttypes];
  double *loc_conc = new double[ttypes];
  double *loc_molfrac = new double[nworlds];

  for(i=0;i<ttypes;i++){
    loc_cnt[i]=0.0;
    glb_cnt[i]=0.0;
    loc_tmp[i]=0.0;
    glb_tmp[i]=0.0;
    mupair[i]=0.0; 
  }


  
  // Gather chem types
  if (me == 0)    
    MPI_Allgather(nchem,ntypes,MPI_DOUBLE,conc_matrix,ntypes,MPI_DOUBLE,roots);
  MPI_Bcast(conc_matrix,ntypes*nworlds,MPI_DOUBLE,0,world);
  inverse(conc_matrix,wmolfrac,init_conc,0.0);
  
  m=1;
  for(i=0;i<nworlds;i++){
    loc_molfrac[i]= wmolfrac[i];
  }


  /*  CHECK AGAINST ANALYTICAL SOLN 
  if(me_universe==0){
    printf(" PRINT CONC MATRIX \n ");
    for(int i=0;i<nworlds;i++){
      printf("i=%d: ",i);
      for(int j=0;j<ntypes;j++){
	printf(" %f ",conc_matrix[i*nworlds+j]);
      }
      printf("\n");
    }
    
    printf(" PRINT INITIAL CONC \n ");
    for(int j=0;j<ntypes;j++)
      printf(" %f ",init_conc[j]);
    printf("\n");
    
    
    printf(" PRINT MOLAR FRACTION \n ");
    for(int j=0;j<nworlds;j++)
      printf(" %f ",wmolfrac[j]);
    printf("\n");
    
    //Binary ONLY    
    printf(" PRINT ANALYTICAL MOLAR FRACTION \n ");
    printf(" %f ",(conc_matrix[2]-init_conc[0])/(conc_matrix[2]-conc_matrix[0]));
    printf(" %f ",(-conc_matrix[0]+init_conc[0])/(conc_matrix[2]-conc_matrix[0]));
    printf("\n");
    
  }
	    
  */
   
  for(i=0;i<ncycles;i++){
    k=pick_random_atom();   
    num_selected=0;
    
    if (k >= 0) {
      ktype = atom->type[k];  
      num_selected++;
      ftype=ktype;
      while(ktype==ftype)
	ftype = (ranboltz->uniform()*ntypes)+1;
    }
    
    
    MPI_Allreduce(&num_selected, &num_selected_all, 1, MPI_INT, MPI_SUM,world);
    num_selected=num_selected_all;
    
    // check mulitple selections 
    if( k>0 && num_selected_all != 1){
      printf("WARNING  selected more than one (%d) particle to flip \n ",num_selected_all);
      continue;
    }
    

    //change type
    if(k>=0){
      atom->type[k] = ftype;

      // estimate molar fraction in each world
      for(j=0;j<ttypes;j++){
	if(j==(iworld*nworlds+ktype))
	  loc_conc[j]=conc_matrix[j]-1.0;
	else if(j==(iworld*nworlds+ftype))
	  loc_conc[j]=conc_matrix[j]+1.0;
	else
	  loc_conc[j]=conc_matrix[j]-1.0;
      }
      
      inverse(loc_conc,wmolfrac,init_conc,0.0);
      
      
      m++;
      for(j=0;j<nworlds;j++)
	loc_molfrac[j] += wmolfrac[j];
    }


    // energy difference
    pe=energy_full(1);
    delta_energy = (pe-energy_stored)/(force->boltz*set_temp[iworld]);

    if(k>=0){    
      index=(ktype-1)*ntypes+(ftype-1);    
      loc_cnt[index]++;
      ratio = nchem[ktype-1]/(nchem[ftype-1]+1.0);
      loc_tmp[index] += exp(-delta_energy)*ratio;
      atom->type[k] = ktype;
    }
    //energy_full(1);

  }
  
  for(j=0;j<nworlds;j++)
    loc_molfrac[j] = (double)loc_molfrac[j]/m;

  MPI_Allreduce(loc_cnt,glb_cnt,ttypes,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(loc_tmp,glb_tmp,ttypes,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(loc_molfrac,wmolfrac,nworlds,MPI_DOUBLE,MPI_SUM,roots);


  for(j=0;j<nworlds;j++)
    wmolfrac[j] = (double)wmolfrac[j]/nworlds;
  
  // combine diagonal information to make the matrix symmetric
  int ij=0,ji=0;
  for(i=0;i<ntypes;i++)
    for(j=i+1;j<ntypes;j++){
      ij = i*ntypes+j;
      ji = j*ntypes+i;
      
      if( (glb_cnt[ij]>0) && (glb_cnt[ji]>0) ){
	mupair[ij]=(-glb_cnt[ij]*log(glb_tmp[ij]/glb_cnt[ij])+glb_cnt[ji]*log(glb_tmp[ji]/glb_cnt[ji]))/(glb_cnt[ij]+glb_cnt[ji])*force->boltz*set_temp[iworld];
	mupair[ji]=-mupair[ij];     
      }else if(glb_cnt[ji]>0){
	mupair[ij]=log(glb_tmp[ji]/glb_cnt[ji])*force->boltz*set_temp[iworld];
	mupair[ji]=-mupair[ij];     
      }else if(glb_cnt[ij]>0){
	mupair[ij]=-log(glb_tmp[ij]/glb_cnt[ij])*force->boltz*set_temp[iworld];
	mupair[ji]=-mupair[ij];     
      }
      
    }

  // Get the ideal change in chemical potential for a gas 
  /*
  double lambda0 = sqrt(force->hplanck*force->hplanck/
		       (2.0*MY_PI*atom->mass[type_list[0]]*force->mvv2e*force->boltz*set_temp[iworld]));

  double lambda1 = sqrt(force->hplanck*force->hplanck/
		       (2.0*MY_PI*atom->mass[type_list[1]]*force->mvv2e*force->boltz*set_temp[iworld]));


  double id_dmu = force->boltz*set_temp[iworld]*3.0*(log(lambda0)-log(lambda1));
  
			 */
  

  free( (void*)loc_cnt );
  free( (void*)loc_conc );
  free( (void*)loc_tmp );
  free( (void*)glb_cnt );
  free( (void*)glb_tmp );
  
}



/* 
   attempt particle exchange between replicas
   Note:  This part of the code was not fully tested. 
*/
int GibbsMultiReplica::attempt_exchange(int pair[],int partner,double world_temp)
{

#ifdef DEBUG1
  printf("EXCHANGE  me=%d: \n",me_universe);
#endif


  
  //make sure replicas are not empty
  int nxchg_partner;
  if(me==0){ 
    MPI_Send(&nxchg,1,MPI_INT,partner,0,universe->uworld);
    MPI_Recv(&nxchg_partner,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
  }
  MPI_Bcast(&nxchg_partner,1,MPI_INT,0,world);

  if (nxchg<=1 || nxchg_partner<=1)
    return 0;
  

  // set bounds
  xlo = domain->boxlo[0];
  xhi = domain->boxhi[0];
  ylo = domain->boxlo[1];
  yhi = domain->boxhi[1];
  zlo = domain->boxlo[2];
  zhi = domain->boxhi[2];
  if (domain->triclinic) {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  } else {
    sublo = domain->sublo;
    subhi = domain->subhi;
  }


  // pick random particle to delete
  int k = -1;
  int insert_type=0;
  int delete_type=0;
  int num_selected=0,num_partner;
  double coord[3];
  double vels[3];
  
  k=pick_random_atom();   
  if(iworld==pair[0])
    k=-1;

  if (k >= 0) {
    delete_type = atom->type[k];  
    double **x = atom->x;
    double **v = atom->v;
    coord[0]=x[k][0];
    coord[1]=x[k][1];
    coord[2]=x[k][2];
    vels[0]=v[k][0];
    vels[1]=v[k][1];
    vels[2]=v[k][2];
    // could add charge attribute as well 
    num_selected++;
  }
  
  int num_selected_all;
  MPI_Allreduce(&num_selected, &num_selected_all, 1, MPI_INT, MPI_SUM,world);
  num_selected=num_selected_all;
  
  if(me==0){ // only root processor participate 
    MPI_Send(&num_selected,1,MPI_INT,partner,0,universe->uworld);
    MPI_Recv(&num_partner,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
  }
  
  // communicate to world
  MPI_Bcast(&num_partner,1,MPI_INT,0,world);

  int num_tot = num_selected+num_partner;
  // check for empty box 
  if( num_tot != 1){
    printf("WARNING  %d+%d !=1 for me= %d part=%d  with k=%d \n",num_selected,num_partner,me_universe, partner, k);
    return 0;
  }

    
  int delete_type_all;
  MPI_Allreduce(&delete_type, &delete_type_all, 1, MPI_INT, MPI_SUM,world);
  delete_type=delete_type_all;

  if(me==0){ 
    MPI_Send(&delete_type,1,MPI_INT,partner,0,universe->uworld);
    MPI_Recv(&insert_type,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
  }
  MPI_Bcast(&insert_type,1,MPI_INT,0,world);

  double vol,vol_partner;
  int    natm,natm_partner;
  
  //  pe_compute->compute_scalar();    
  //  pe = energy_full(1); 

  vol = domain->xprd * domain->yprd * domain->zprd;
  natm = atom->natoms;

  // only root processor participate in decision making logic
  if(me==0){ // communicate partner vol and number of atoms
    MPI_Send(&vol,1,MPI_DOUBLE,partner,0,universe->uworld);
    MPI_Send(&natm,1,MPI_INT,partner,0,universe->uworld);
    MPI_Recv(&vol_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    MPI_Recv(&natm_partner,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
  }


  double energy_before = 0,energy_after = 0;
  double pe,pe_partner;


  pe = energy_stored; 
  
  // hi proc sends PE to low proc
  // lo proc makes the sum and communicates it back to hi proc    
  if(me==0){ 
    if (me_universe > partner)
      MPI_Send(&pe,1,MPI_DOUBLE,partner,0,universe->uworld);
    else
      MPI_Recv(&pe_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    if (me_universe < partner){
      energy_before = pe + pe_partner;
      MPI_Send(&energy_before,1,MPI_DOUBLE,partner,0,universe->uworld);
    }else
      MPI_Recv(&energy_before,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
  }
  
  
  // STORE STATES 
  store_state();

  // attempt insertion first
  int insert_success=0,delete_success=0,metro_success=0;
  int proc_flag = 0;
  // attempt insertion in selected world by root
  //  if(0){  
  if(me==0 && insert_type>0){  
    double lamda[3];
    //  ninsertion_attempts += 1.0;
    if (domain->triclinic == 0) {
      coord[0] = xlo + ranboltz->uniform() * (xhi-xlo);
      coord[1] = ylo + ranboltz->uniform() * (yhi-ylo);
      coord[2] = zlo + ranboltz->uniform() * (zhi-zlo);
    } else {
      lamda[0] = ranboltz->uniform();
      lamda[1] = ranboltz->uniform();
      lamda[2] = ranboltz->uniform();
      
      // wasteful, but necessary                                                                        
      if (lamda[0] == 1.0) lamda[0] = 0.0;
      if (lamda[1] == 1.0) lamda[1] = 0.0;
      if (lamda[2] == 1.0) lamda[2] = 0.0;
      
      domain->lamda2x(lamda,coord);
    }
    

    if (domain->triclinic == 0) {
      domain->remap(coord);
      if (!domain->inside(coord))
	printf("gibbs/multireplica put atom outside box \n");
      //	  error->one(FLERR,"gibbs/multireplica put atom outside box");
      
      if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
	  coord[1] >= sublo[1] && coord[1] < subhi[1] &&
	  coord[2] >= sublo[2] && coord[2] < subhi[2]) proc_flag = 1;
    } else {
      if (lamda[0] >= sublo[0] && lamda[0] < subhi[0] &&
	  lamda[1] >= sublo[1] && lamda[1] < subhi[1] &&
	  lamda[2] >= sublo[2] && lamda[2] < subhi[2]) proc_flag = 1;
    }
    
    if (proc_flag) {
      atom->avec->create_atom(insert_type,coord);
      int m = atom->nlocal - 1;

      // add to groups      
      // optionally add to type-based groups                                                          
      
      atom->mask[m] = groupbitall;
      /* //ignore this part for now
	 for (int igroup = 0; igroup < ngrouptypes; igroup++) {
	 if (partner_type == grouptypes[igroup])
	 atom->mask[m] |= grouptypebits[igroup];
	 }
      */
      atom->v[m][0] = vels[0];//ranboltz->gaussian()*sigma;
      atom->v[m][1] = vels[1];//ranboltz->gaussian()*sigma;
      atom->v[m][2] = vels[1];//ranboltz->gaussian()*sigma;
      modify->create_attribute(m);
      insert_success = 1;
    }
  }
  
  int insert_success_all=0,insert_success_partner=0;
  MPI_Allreduce(&insert_success, &insert_success_all, 1, MPI_INT, MPI_SUM,world);

  if(me==0){
    MPI_Send(&insert_success_all,1,MPI_INT,partner,0,universe->uworld);
    MPI_Recv(&insert_success_partner,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
  }
  MPI_Bcast(&insert_success_partner,1,MPI_INT,0,world);
  

  if((insert_success_all+insert_success_partner)==0){
    return 0;
  }
  

  if (insert_success_all) {
    atom->natoms++;   // create_atoms takes care nlocal++
    //    ninsertion_successes += 1.0;
  }
  
  int tmpmask;
  if(k >=0 && insert_success_partner){
    //  if(k >=0){
    tmpmask = atom->mask[k];
    atom->avec->copy(atom->nlocal-1,k,1);
    atom->nlocal--;
    delete_success=1;
  }
  
  int delete_success_all=0;
  MPI_Allreduce(&delete_success, &delete_success_all, 1, MPI_INT, MPI_SUM,world);
  

  if (delete_success_all) {
    atom->natoms--;
    //    ndeletion_successes += 1.0;
  }
  
  
  // communicate 
  if (atom->tag_enable) {
    if (atom->map_style){ 
      atom->map_init();
      atom->map_set();
    }
  }
  atom->nghost = 0;
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  


  //gather energy after 
  pe = energy_full(1); 
  
  // only root processor participate
  // processor with smaller (global) rank decides whether accept or not
  if(me==0){ 
    if (me_universe > partner) 
      MPI_Send(&pe,1,MPI_DOUBLE,partner,0,universe->uworld);
    else
      MPI_Recv(&pe_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    
    if(me_universe < partner){
      energy_after = pe + pe_partner;
      if(ranboltz->uniform() <
	 exp((energy_before-energy_after)/(force->boltz*world_temp))-(vol_partner/vol)*(natm+1)/(natm_partner))
	metro_success=1;
      else
	metro_success=0;
      MPI_Send(&metro_success,1,MPI_INT,partner,0,universe->uworld);
    }else
      MPI_Recv(&metro_success,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
  }
 
  // synchronize success results via allreduce to the world
  int metro_success_all = 0;
  MPI_Allreduce(&metro_success,&metro_success_all,1,MPI_INT,MPI_MAX,world);

  if(!metro_success_all){
    if (insert_success_all){
      atom->natoms--;
      if (proc_flag) 
	atom->nlocal--;
      revert_state();
    }

    if(delete_success_all)
      atom->natoms++;
    if(k>=0){
      atom->mask[k] = tmpmask;
      atom->avec->create_atom(delete_type,coord); //   atom->nlocal++ taken care by create_atom
      int m = atom->nlocal - 1;
      atom->mask[m] = groupbitall;
      atom->v[m][0] = vels[0];
      atom->v[m][1] = vels[1];
      atom->v[m][2] = vels[2];
      modify->create_attribute(m);
    }

    
    //    printf("REVERTING  me= %d part=%d  (%d,%d) \n",me_universe, partner, atom->natoms,atom->nlocal);

    pe = energy_full(1); 
    
    update_all_atoms_list();
    return 0;

  }

  update_all_atoms_list();
  return 1;
}

/* ----------------------------------------------------------------------

------------------------------------------------------------------------- */



int GibbsMultiReplica::attempt_inter_swap(int world_pair[],int partner,double world_temp,int swap_type[])
{

#ifdef DEBUG1
  printf("INTER_SWAP  me=%d: \n",me_universe);
#endif
  
  // count chemistry before swap
  count_chemistry();  
  
  // pick random k atom from list of atoms to swap
  int k = -1;
  //i is the type of world_pair[0] world
  //j is the type of world_pair[1] world

  int ktype=-1;
  int ptype=-1;


  double ratio =0,ratio_partner=0;

  if(iworld == world_pair[1]){       
    if (niswap > 0)
      k=pick_i_swap_atom();
    ktype=swap_type[0];
    ptype=swap_type[1];
    ratio = nchem[ktype-1]/(nchem[ptype-1]+1.0);
  }else if(iworld == world_pair[0]){
    if (njswap > 0)
      k=pick_j_swap_atom();
    ktype=swap_type[1];
    ptype=swap_type[0];
    ratio = nchem[ktype-1]/(nchem[ptype-1]+1.0);
  }else{
    //    printf("WARNING!!! INTER  me_universe %d  partner = %d  and  world1=%d or world2=%d  \n",me_universe,partner,world_pair[0],world_pair[1]);
  }
  


   
  int num_selected=0,num_partner=0,num_pair=0;
  double boltz_factor,sqrt_factor;
  if (k >= 0) {
    if(ktype != atom->type[k])
      printf("WARNING!!! inter-type in list is inconsistent \n");
    //      error->all(FLERR,"type in list is inconsistent");      
    atom->type[k] = ptype;  // partner type
    sqrt_factor=sqrt(atom->mass[ktype]/atom->mass[ptype]);
    num_selected++;
  }


  // if pairing is successful 
  // attempt swap otherwise return
  // Reduce all of the local sums into the global sum
  int num_selected_all;
  MPI_Allreduce(&num_selected, &num_selected_all, 1, MPI_INT, MPI_SUM,world);
  /*
  if (num_selected_all!=1){
    printf("WARNING!!! inside inter swap, there are %d !=1  selected particles. \n",num_selected_all);
    //    error->all(FLERR,"more than one particle selected");
    }
  */
  num_selected=num_selected_all;
  
  if (partner != -1){
    if(me==0){ 
      MPI_Send(&num_selected,1,MPI_INT,partner,0,universe->uworld);
      MPI_Recv(&num_partner ,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    }

    if(me==0){ 
      MPI_Send(&ratio,1,MPI_DOUBLE,partner,0,universe->uworld);
      MPI_Recv(&ratio_partner ,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    }
  
  }
  // communicate to world
  MPI_Bcast(&num_partner,1,MPI_INT,0,world);

  // check for matching pairs
  int matchfails=0;
  if(iworld == world_pair[0] || iworld == world_pair[1])
    if(num_selected != 1 || num_partner != 1){
      matchfails=1;
    }

  // communicate to roots
  int matchfails_all;
  MPI_Allreduce(&matchfails,&matchfails_all,1,MPI_INT,MPI_MAX,roots);  
  
  if(matchfails_all){
    if(k>=0)
      atom->type[k] = ktype;  
    return 0;
  }
  
    
  double *save_frac= new double[nworlds];  
  for(int i=0;i<nworlds;i++)
    save_frac[i]= molar_fraction[i];    
  
  double energy_before = 0,energy_after = 0;
  double pe,pe_partner; 
  int success=0;
  
  pe=energy_stored;
  
  
  // hi proc sends PE to low proc
  // lo proc makes the sum and communicates it back to hi proc    
  if (partner != -1) 
    if(me==0){ // only root processor participate in decision making logic
      if (me_universe > partner) 
	MPI_Send(&pe,1,MPI_DOUBLE,partner,0,universe->uworld);
      else
	MPI_Recv(&pe_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
      if (me_universe < partner){
	energy_before = pe + pe_partner;
	MPI_Send(&energy_before,1,MPI_DOUBLE,partner,0,universe->uworld);
      }else
	MPI_Recv(&energy_before,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    }
  
  //update variables 
  //update_all_atoms_list();

  //evaluate energy and chemistry after swap
  count_chemistry();  
  pe = energy_full(1);
  
  // only root processor participate
  // processor with smaller (global) rank decides whether accept or not
  if (partner != -1) 
    if(me==0){ 
      if (me_universe > partner) 
	MPI_Send(&pe,1,MPI_DOUBLE,partner,0,universe->uworld);
      else
	MPI_Recv(&pe_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
      
      if(me_universe < partner){
	energy_after = pe + pe_partner;
	if(ranboltz->uniform() < ratio*ratio_partner*exp((energy_before-energy_after)/(force->boltz*world_temp))) 
	  success=1;
	else
	  success=0;
	MPI_Send(&success,1,MPI_INT,partner,0,universe->uworld);
      }else
	MPI_Recv(&success,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    }
  
  // synchronize success results via allreduce to the world
  int success_all = 0;
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);
  
  if(k>=0){
#ifdef DEBUG2
    printf("inter world %d:%d , particle selected is %d of type %d->%d, success ? %d \n",world2root[iworld],partner,i,itype,jtype,success);
#endif
  }

    
  // Check that molar fraction has a solution
  if (me == 0)    
    MPI_Allgather(nchem,ntypes,MPI_DOUBLE,conc_matrix,ntypes,MPI_DOUBLE,roots);
  MPI_Bcast(conc_matrix,ntypes*nworlds,MPI_DOUBLE,0,world);
  inverse(conc_matrix,molar_fraction,init_conc,0.0);

  
  int invflag=1;
  // skip if molar fraction cannot be found
  if(!check_molfrac(molar_fraction)){
    invflag=0;
    if(me==0){
      printf("inter-swap attempt inconsitent with molar fraction \n");
      for(int k=0;k<nworlds;k++)
	printf(" f[%d] =  %f ",k,molar_fraction[k]);
      printf("\n");
    }
  }
  
  MPI_Bcast(&invflag,1,MPI_INT,0,universe->uworld);
  
  if(slopeflag && invflag==0){ 
    for(int i=0;i<nworlds;i++)
      molar_fraction[i]=save_frac[i];
    success_all=0;
  }
  
  free( (void*)save_frac );
    
  if (success_all) {
    energy_stored = pe;

    /* conserve the kinetic energy */
    if (k >= 0) {   
      atom->v[k][0] *= sqrt_factor;
      atom->v[k][1] *= sqrt_factor;
      atom->v[k][2] *= sqrt_factor;
    }
    return 1;
  }else{
    if (k >= 0){
#ifdef DEBUG2
      printf("inter world %d:%d , remake particle %d of type %d->%d \n",world2root[iworld],partner,i,jtype,itype);
#endif
      atom->type[k] = ktype;
    }

   energy_full(1);

  }

  return 0;
}


/* 
   attempt a number of particle flips
*/

int GibbsMultiReplica::attempt_flip(int iswap, int world_pair[],int partner,double world_temp,double dmu)
{

#ifdef DEBUG1
  printf("FLIP  me=%d: \n",me_universe);
#endif

  // allocate array storing the indices of particles to be flipped
  if (atom->nmax > atom_many_nmax) {
    memory->sfree(local_katoms_list);
    atom_many_nmax = 3*atom->nmax;
    local_katoms_list = (int *) memory->smalloc(atom_many_nmax*sizeof(int),
                                                "GIBBSMULTIREPL:local_katoms_list");
  }

  for (int i = 0; i < atom_many_nmax; i++) {
    local_katoms_list[i] = -1;
  }

  // pick random particle to flip 
  int k = -1,which=-1;
  int ktype=0;  // type of random particle 
  int ftype=-1; // flip to this type
  int num_selected=0;
  double sqrt_factor;

  int nflip = flip_nmax;
  //int nflip = flip_nmax*random_equal->uniform()+1;
  //int nflip = flip_nmax+4-5*random_equal->uniform();
  if(nflip<1) nflip=1;
  double arr[ntypes],dn[ntypes],dn_partner[ntypes];  
  for(int i=0;i<ntypes;i++){
    arr[i]=0.0;
    dn[i]=0.0;
    dn_partner[i]=0.0;
    mu_vect[i]=0.0;
  }

  const int ttypes = ntypes*ntypes;
  double *arr_flip = new double[ttypes];
  double *pair_flip = new double[ttypes];
  double *pair_partner = new double[ttypes];


  for(int i=0;i<ttypes;i++){
    arr_flip[i]=0.0;
    pair_flip[i]=0.0;
    pair_partner[i]=0.0;
    mu_matrix[i]=0.0;
  }
  
  int num_selected_all=0;
  int index=0;

  if(iworld==world_pair[0]){  // attempt flip on world[0]  
    while(num_selected_all<nflip){
      k=pick_random_atom();   
      if (k >= 0) {
	ktype = atom->type[k];  
	arr[ktype-1]--;
	ftype=ktype;
	while(ktype==ftype)
	  ftype = (ranboltz->uniform()*ntypes)+1; 
	index = num_selected*3;
	local_katoms_list[index]=k;
	local_katoms_list[index+1]=ktype;
	local_katoms_list[index+2]=ftype;
	num_selected++;   
	arr[ftype-1]++;
	arr_flip[(ftype-1)*ntypes+(ktype-1)]--;
	arr_flip[(ktype-1)*ntypes+(ftype-1)]++;
      }
      MPI_Allreduce(&num_selected, &num_selected_all, 1, MPI_INT, MPI_SUM,world);
    }
  }
  

  num_selected=num_selected_all;
  // check mulitple selections 
  if( (iworld==world_pair[0]) && (num_selected_all != nflip) ){
    printf("WARNING  selected particles do not correspond to number of particles to flip %d<->%d \n ",nflip,num_selected);
    return 0;
  }
  
  // gather arrays info from each world
  MPI_Allreduce(arr,dn,ntypes,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(arr_flip,pair_flip,ttypes,MPI_DOUBLE,MPI_SUM,world);

  /*  DEBUG
  if(me_universe==0){
    printf("Printing flip pair \n");
    for (int i = 0; i < ntypes; i++){
      for (int j = 0; j < ntypes; j++)
	printf(" %f ",pair_flip[i*ntypes+j]);
      printf("\n");
    }

    printf("Print dn  \n");
    for (int j = 0; j < ntypes; j++)
      printf(" %f ",dn[j]);
    printf("\n");

  }
  */


  double sum=0.0,sum2=0.0;
  int skipflag=0;
  for(int i=0;i<ntypes;i++){
    sum +=dn[i];
  }


  if(sum!=0.0)
    printf("WARNING species changes are not conserved sum=%f \n ",sum);

  double vol,vol_partner;
  vol = domain->xprd * domain->yprd * domain->zprd;

  // communicate array info between worlds
  if(partner != -1){
    if(me==0){ 
      MPI_Send(dn,ntypes,MPI_DOUBLE,partner,0,universe->uworld);
      MPI_Recv(dn_partner,ntypes,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    }
    if(me==0){ 
      MPI_Send(pair_flip,ttypes,MPI_DOUBLE,partner,0,universe->uworld);
      MPI_Recv(pair_partner,ttypes,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    }

    if(me==0){
      MPI_Send(&vol,1,MPI_DOUBLE,partner,0,universe->uworld);
      MPI_Recv(&vol_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
    }
  }


  int success=0;
  double wene_before = 0,wene_after = 0;
  double pe= energy_stored,wpe;  // molar weighted energies
  double tmp,tmp_partner,conc_ct,conc_lev;
  double sconfig=0.0;
  double da=0.0,db=0.0,dbb=0.0;

  double *save_frac= new double[nworlds];  
  for(int i=0;i<nworlds;i++)
    save_frac[i]= molar_fraction[i];    
  
  if (me == 0)    
    MPI_Allgather(nchem,ntypes,MPI_DOUBLE,conc_matrix,ntypes,MPI_DOUBLE,roots);
  MPI_Bcast(conc_matrix,ntypes*nworlds,MPI_DOUBLE,0,world);
  inverse(conc_matrix,molar_fraction,init_conc,0.0);

  
  int invflag=1;
  // skip if molar fraction cannot be found
  if(!check_molfrac(molar_fraction))
    invflag=0;
  
  MPI_Bcast(&invflag,1,MPI_INT,0,universe->uworld);

  if(invflag==0){ 
    for(int i=0;i<nworlds;i++)
      molar_fraction[i]=save_frac[i];
    
    if (me == 0)
      printf(" NO SOLN found during FLIP \n ");
  
    return 0;
  }

  /*
  if(me_universe==0){
    printf("Printing mu matrix \n");
    for (int i = 0; i < ntypes; i++){
      for (int j = 0; j < ntypes; j++)
	printf(" %2.4f ",mu_matrix[i*ntypes+j]);
      printf("\n");
    }

    for(int i=0;i<nworlds;i++)
      printf(" %2.3f ",mu_vect[i]);
    printf("\n");    
   
  }
  */
  inverse(mu_matrix,mu,mu_vect,0.0);
  /*
  if(me_universe==0){
    for(int i=0;i<nworlds;i++)
      printf(" %2.3f ",mu[i]);
    printf("\n");
  }
  */

  /////////////////////////
  if(iworld==world_pair[1]){ // this correspond to coupled-world where no particles where flippled
    for(int i=0;i<ntypes;i++)
      dn[i]=-(molar_fraction[partner]/molar_fraction[iworld])*dn_partner[i]; 
    
    for (int i = 0; i < ntypes; i++)
      for (int j = 0; j < ntypes; j++){
	index=i*ntypes+j;
	pair_flip[index]=-(molar_fraction[partner]/molar_fraction[iworld])*pair_partner[index]; 
      }
  }
  /////////////////////////

  // common tangent prediction                                                                      
  db=0.0;
  for (int i = 0; i < ntypes; i++)
    for (int j = 0; j < ntypes; j++){
      index=i*ntypes+j;
     db +=muavg[index]*pair_flip[index]*molar_fraction[iworld]/2.0;
    }
  
  double BW=0.0,BV=0.0,BG=0.0; // Braggs-Williams term
  for(int k=0;k<ntypes;k++){
    if(nchem[k]>0)
      BW -= Ntot*molar_fraction[iworld]*(nchem[k]/nxchg)*log(Ntot*molar_fraction[iworld]*(nchem[k]/nxchg));
  }


  if( molar_fraction[iworld]>0)
    BV = Ntot*molar_fraction[iworld]*log(Ntot*molar_fraction[iworld]*vol/nxchg);
  
  sconfig = (BW+BV)*force->boltz*world_temp/nworlds;
  BG = Ntot*(pe/nxchg)*molar_fraction[iworld];
    
  // first part of weighted sum
  wpe = (1.0-fudge)*(BG-sconfig);
 
  double *wene = new double[nworlds];  
  if (me == 0)    // gather new info
    MPI_Allgather(&wpe,1,MPI_DOUBLE,wene,1,MPI_DOUBLE,roots);        
  MPI_Bcast(wene,nworlds,MPI_DOUBLE,0,world);

 
  for(int i=0;i<nworlds;i++){
    wene_before += wene[i]*nworlds; 
  }

  // Change particles type
  for(int i=0;i<num_selected_all;i++){
    index=3*i;
    k=local_katoms_list[index];
    ftype=-1;
    if(k>=0){
      ftype=local_katoms_list[index+2];
      atom->type[k] = ftype;                                                                              
    }
  }

  //update variables 
  // update_all_atoms_list(); 
  count_chemistry();  

  vol = domain->xprd * domain->yprd * domain->zprd;
  pe = energy_full(1);

  if (me == 0)    
    MPI_Allgather(nchem,ntypes,MPI_DOUBLE,conc_matrix,ntypes,MPI_DOUBLE,roots);
  MPI_Bcast(conc_matrix,ntypes*nworlds,MPI_DOUBLE,0,world);

  inverse(conc_matrix,molar_fraction,init_conc,0.0);


  // common tangent prediction                                                                      
  da=0.0;
  for (int i = 0; i < ntypes; i++)
    for (int j = 0; j < ntypes; j++){
      index=i*ntypes+j;
      da +=muavg[index]*pair_flip[index]*molar_fraction[iworld]/2.0;
    }
  
  
  BW=0.0,BV=0.0,BG=0.0; // Braggs-Williams term
  for(int k=0;k<ntypes;k++){
    if(nchem[k]>0)
      BW -= Ntot*molar_fraction[iworld]*(nchem[k]/nxchg)*log(Ntot*molar_fraction[iworld]*(nchem[k]/nxchg));
  }
  
  if( molar_fraction[iworld]>0)
    BV = Ntot*molar_fraction[iworld]*log(Ntot*molar_fraction[iworld]*vol/nxchg);
  
  sconfig = (BW+BV)*force->boltz*world_temp/nworlds;
  BG = Ntot*(pe/nxchg)*molar_fraction[iworld];

  // second part of weighted sum
  wpe = (1.0-fudge)*(BG-sconfig)+fudge*(db+da)/2.0;  

  if (me == 0)    // gather new info
    MPI_Allgather(&wpe,1,MPI_DOUBLE,wene,1,MPI_DOUBLE,roots);        
  MPI_Bcast(wene,nworlds,MPI_DOUBLE,0,world);
  
  
  for(int i=0;i<nworlds;i++){
    wene_after += wene[i]*nworlds; 
  }
  // only root processor participate
  // processor with smaller (global) rank decides whether accept or not
  if (partner != -1) 
    if(me==0){ 
      if(me_universe < partner){
	if(ranboltz->uniform() <
	   exp((-wene_after+wene_before)/(force->boltz*world_temp)))
	  success=1;
	else
	  success=0;
	MPI_Send(&success,1,MPI_INT,partner,0,universe->uworld);
      }else{
	MPI_Recv(&success,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);
      }
      
  }
  
  // synchronize success results via allreduce to the world
  int success_all = 0;
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);

  if(me==0){ 
    char s[100] = {0};
    sprintf(s, "FlipStats_prc%d.txt", me_universe); 
    FILE *fp;
    fp = fopen(s, "a+");
    fprintf(fp," %d %d %f %f %f %f %f %f %d \n",iswap,success_all,world_temp,molar_fraction[iworld],pe,mu[0],mu[1],mu_matrix[0]*mu[0]+mu_matrix[2]*mu[1],nflip);
    fclose(fp);
  }  

  invflag=1;
  if(!check_molfrac(molar_fraction))
    invflag=0;
  
  MPI_Bcast(&invflag,1,MPI_INT,0,universe->uworld);

  if(invflag==0)
    success_all=false;

  if (success_all) {
    energy_stored = pe;
    /* conserve the kinetic energy */

    for(int i=0;i<num_selected_all;i++){
      index=3*i;
      k=local_katoms_list[index];
      if(k>=0){
	ktype=local_katoms_list[index+1];
	ftype=local_katoms_list[index+2];
	sqrt_factor=sqrt(atom->mass[ktype]/atom->mass[ftype]);
	//	atom->v[k][0] /= sqrt_factor;
	//	atom->v[k][1] /= sqrt_factor;
	//	atom->v[k][2] /= sqrt_factor;
      }
    }
  
  }else{

    for(int i=0;i<nworlds;i++)
      molar_fraction[i]=save_frac[i];    
    
    for(int i=0;i<num_selected_all;i++){
      index=3*i;
      k=local_katoms_list[index];
      if(k>=0){
        ktype=local_katoms_list[index+1];
	atom->type[k] = ktype;
      }
    }

   energy_full(1);
  }


  // release memory
  free( (void*)arr_flip );
  free( (void*)pair_flip );
  free( (void*)pair_partner );
  free( (void*)wene );
  free( (void*)save_frac );

  return success_all;


}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int GibbsMultiReplica::attempt_intra_swap(int types[],double world_temp)
{


#ifdef DEBUG1
  printf("INTRA  me=%d: \n",me_universe);
#endif

  
  if ( (niswap == 0) || (njswap == 0) ){
    //    error->all(FLERR,"species to swap has not been selected");
    return 0;
  }

  double energy_before = 0,energy_after = 0;
  double boltz_factor,sqrt_factor;
  int success=0;

  energy_before = energy_stored;

  int i = pick_i_swap_atom();
  int j = pick_j_swap_atom();
  int itype = types[0];
  int jtype = types[1];
  int num_selected=0;

  sqrt_factor=sqrt(atom->mass[itype]/atom->mass[jtype]);

  if (i >= 0) {
    if(itype != atom->type[i]){
      printf("WARNING!!! type in i-list is inconsistent \n");
      printf("pre-intra-j world %d , particle selected is %d of type %d->%d \n",world2root[iworld],j,itype,jtype);
    //      error->all(FLERR,"type in i-list is inconsistent");      
    }else{
      num_selected++;
    }
  }

  if (j >= 0){
    if(jtype != atom->type[j]){
      printf("WARNING!!! type in j-list is inconsistent \n");
      printf("pre-intra-j world %d , particle selected is %d of type %d->%d \n",world2root[iworld],j,itype,jtype);
    //error->all(FLERR,"type in j-list is inconsistent");      
    }else{
      num_selected++;
    }
  }

    
  // Reduce all of the local sums into the global sum
  int num_selected_all;
  MPI_Allreduce(&num_selected, &num_selected_all, 1, MPI_INT, MPI_SUM,world);
  if (num_selected_all!=2){
    printf("did not find pair  %d:  %d ,%d , %d \n",me_universe,i,j,num_selected_all);
    return 0;
  }else{
    if (i >= 0) 
      atom->type[i] = jtype;
    if (j >= 0) 
      atom->type[j] = itype;  
  }

  // compute energy after swap
  energy_after = energy_full(1); 
  
  if(me==0)
    if(ranboltz->uniform() < 
       exp((energy_before-energy_after)/(force->boltz*world_temp)))                                                    
      success=1;
  
  // bcast result to other procs in this world                                                                                    
  //  MPI_Bcast(&success,1,MPI_INT,0,world);  
  // synchronize results 
  int success_all = 0;
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);


#ifdef DEBUG222
  if(i>=0)
    printf("intra-i world %d , particle selected is %d of type %d->%d success ? %d \n",world2root[iworld],i,itype,jtype,success);
  else if(j>=0)
    printf("intra-j world %d , particle selected is %d of type %d->%d success ? %d \n",world2root[iworld],j,itype,jtype,success);
#endif

#ifdef DEBUG222
  printf("INTRA SWAP inside world %d with success?%d:%d  and dE=%f \n",me_universe,success,success_all,energy_after-energy_before);
#endif


  if (success_all) {
    energy_stored = energy_after;
    
    /* conserve the kinetic energy */
    if (i >= 0) {   
      atom->v[i][0] *= sqrt_factor;
      atom->v[i][1] *= sqrt_factor;
      atom->v[i][2] *= sqrt_factor;
    }

    if (j >= 0) {   
      atom->v[j][0] /= sqrt_factor;
      atom->v[j][1] /= sqrt_factor;
      atom->v[j][2] /= sqrt_factor;
    }

    return 1;
  } else {

    if (i >= 0) {
      atom->type[i] = itype;
    }
    
    if (j >= 0) {
      atom->type[j] = jtype;
    }

   energy_full(1);
  }
  return 0;
  
}








/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int GibbsMultiReplica::pick_random_atom()
{

  int i = -1;
  int iwhichglobal = static_cast<int> (nxchg*random_equal->uniform());

  if ((iwhichglobal >= nxchg_before) &&
      (iwhichglobal < nxchg_before + nxchg_local)) {
    int iwhichlocal = iwhichglobal - nxchg_before;
    i = local_all_atoms_list[iwhichlocal];
  }


  
  return i;
  
}


/* ----------------------------------------------------------------------                                                                             
   ------------------------------------------------------------------------- */

int GibbsMultiReplica::pick_i_swap_atom()
{

  // This function does not quite work when the partition is as n x m where m is an odd number

  int i = -1;
  int iwhichglobal = static_cast<int> (niswap*random_equal->uniform());
  if ((iwhichglobal >= niswap_before) &&
      (iwhichglobal < niswap_before + niswap_local)) {
    int iwhichlocal = iwhichglobal - niswap_before;
    i = local_swap_iatom_list[iwhichlocal];
  }

  return i;
}

/* ----------------------------------------------------------------------                                                                             
   ------------------------------------------------------------------------- */

int GibbsMultiReplica::pick_j_swap_atom()
{
  int j = -1;
  int jwhichglobal = static_cast<int> (njswap*random_equal->uniform());
  if ((jwhichglobal >= njswap_before) &&
      (jwhichglobal < njswap_before + njswap_local)) {
    int jwhichlocal = jwhichglobal - njswap_before;
    j = local_swap_jatom_list[jwhichlocal];
  }

  return j;
}



/* ----------------------------------------------------------------------
   update the list of all atoms
   ------------------------------------------------------------------------- */

void GibbsMultiReplica::update_all_atoms_list()
{
  
  // this list needs to be updated only if atoms are added or deleted.
  

  int nlocal = atom->nlocal;
  double **x = atom->x;
  
  if (atom->nmax > atom_all_nmax) {
    memory->sfree(local_all_atoms_list);
    atom_all_nmax = atom->nmax;
    local_all_atoms_list = (int *) memory->smalloc(atom_all_nmax*sizeof(int),
						   "GIBBSMULTIREPLICA:local_all_atoms_list");  
  }
  

  //  int bitmask = group->bitmask[firstgroup];
  //    int firstgroupbit = group->bitmask[atom->firstgroup];
  //int igroup = group->find(0); // all group id
  //  if (igroup == -1) error->all(FLERR,"Could not find all group ID");
  //  int groupbit = group->bitmask[igroup];
  //  if (m >= 0 && m < nlocal) {

  
  nxchg_local = 0;
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i]) {   // add bitgroup
	local_all_atoms_list[nxchg_local] = i;
	nxchg_local++;
    }
  }

  MPI_Allreduce(&nxchg_local,&nxchg,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&nxchg_local,&nxchg_before,1,MPI_INT,MPI_SUM,world);
  nxchg_before -= nxchg_local;

  /*
  if(iswap >=0)
    {   
      char s[100] = {0};
      sprintf(s, "file_prc%d_step%d.txt", me_universe,iswap); 
      FILE *fp;
      fp = fopen(s, "w+");
      fprintf(fp, "iswap=%d  me_univ=%d itype=%d  nxchg_local=%d nxchg=%d \n",iswap,me_universe,types[0],nxchg_local,nxchg);
      fclose(fp);
    }
  */
  /*
      int idg = atom->tag[i];
      int idl = atom->map(idg);
      */

}




/* ----------------------------------------------------------------------
   update the list of swap atoms to be exchanged
   ------------------------------------------------------------------------- */

void GibbsMultiReplica::update_swap_atoms_list(int swap_type[])
{
  
  // this list needs to be updated if atoms inside cells change type 

  if ((swap_type[0] <= 0) || (swap_type[1] <=0) ) //error->all(FLERR,"species to swap has not been selected");
    printf("WARNING!!! intra-list type in list is inconsistent \n");  
  int nlocal = atom->nlocal;
  double **x = atom->x;
  
  if (atom->nmax > atom_i_intra_nmax) {
    memory->sfree(local_swap_iatom_list);
    atom_i_intra_nmax = atom->nmax;
    local_swap_iatom_list = (int *) memory->smalloc(atom_i_intra_nmax*sizeof(int),
						    "GIBBSMULTIREPLICA:local_swap_iatom_list");
  }

  if (atom->nmax > atom_j_intra_nmax) {
    memory->sfree(local_swap_jatom_list);
    atom_j_intra_nmax = atom->nmax;
    local_swap_jatom_list = (int *) memory->smalloc(atom_j_intra_nmax*sizeof(int),
						    "GIBBSMULTIREPLICA:local_swap_jatom_list");
  }
  
  //  int bitmask = group->bitmask[firstgroup];
  //  int firstgroupbit = group->bitmask[atom->firstgroup];
  //  int igroup = group->find(0); // all group id
  //  if (igroup == -1) error->all(FLERR,"Could not find all group ID");
  //  int groupbit = group->bitmask[igroup];
  //  if (m >= 0 && m < nlocal) {
 
  
  niswap_local = 0;
  njswap_local = 0;


  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] ) {
      int itype = atom->type[i];
      if (itype == swap_type[0]){
	local_swap_iatom_list[niswap_local] = i;
	niswap_local++;
      }else if (itype == swap_type[1]){
	local_swap_jatom_list[njswap_local] = i;
	njswap_local++;
      }
    }
  }
  
  
  MPI_Allreduce(&niswap_local,&niswap,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&niswap_local,&niswap_before,1,MPI_INT,MPI_SUM,world);
  niswap_before -= niswap_local;
  
  MPI_Allreduce(&njswap_local,&njswap,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&njswap_local,&njswap_before,1,MPI_INT,MPI_SUM,world);
  njswap_before -= njswap_local;
}

/* ----------------------------------------------------------------------                                                        
   compute hydrostatic target pressure                                                                                           
   -----------------------------------------------------------------------*/

void GibbsMultiReplica::compute_press_target() {
  double delta = update->ntimestep - update->beginstep;
  delta /= update->endstep - update->beginstep;
  if (update->endstep > update->beginstep)
    delta /= update->endstep - update->beginstep;
  else delta = 0.0;

  p_hydro = 0.0;
  for (int i = 0; i < 3; i++) {
    set[i].p_target = set[i].p_start + delta * (set[i].p_stop-set[i].p_start);
    p_hydro += set[i].p_target;
  }
  p_hydro /= 3.0; // pdim;

}



/* ----------------------------------------------------------------------
      store state in fix_revert                                                                                         
 ------------------------------------------------------------------------ */

void GibbsMultiReplica::store_state()
{
    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;

    double **astore = fix_revert->astore;

    for (int i = 0; i < nlocal; i++) {
      astore[i][0] = x[i][0];
      astore[i][1] = x[i][1];
      astore[i][2] = x[i][2];
      astore[i][3] = v[i][0];
      astore[i][4] = v[i][1];
      astore[i][5] = v[i][2];
      astore[i][6] = f[i][0];
      astore[i][7] = f[i][1];
      astore[i][8] = f[i][2];
      *((imageint *) &astore[i][9]) = image[i];
    }
}


/* ----------------------------------------------------------------------                                                                restore state archived in fix_revert
   ------------------------------------------------------------------------- */

void GibbsMultiReplica::revert_state()
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  double **astore = fix_revert->astore;

  for (int i = 0; i < nlocal; i++) {
    x[i][0] = astore[i][0];
    x[i][1] = astore[i][1];
    x[i][2] = astore[i][2];
    v[i][0] = astore[i][3];
    v[i][1] = astore[i][4];
    v[i][2] = astore[i][5];
    f[i][0] = astore[i][6];
    f[i][1] = astore[i][7];
    f[i][2] = astore[i][8];
    image[i] = *((imageint *) &astore[i][9]);
  }
}



/* ----------------------------------------------------------------------
   Check the validity of the molar fraction 
   ------------------------------------------------------------------------- */

int GibbsMultiReplica::check_molfrac(double f[])
{
  double sum=0.0;
  
  bool negval =false;
  for (int i = 0; i < nworlds; i++){
    sum +=f[i];
    if(f[i]<0.0)
      return 0;
  }
  
    
  //if(fabs(sum-1.0)>.015)
  if(sum>1.01)
    return 0;
  else
    return 1;
  //  return 1;
}



/* ----------------------------------------------------------------------
   initialize multireplicas with pure elements 
   ------------------------------------------------------------------------- */

void GibbsMultiReplica::set_world_chemistry(int types[])
{

   if (types[0] <= 0) error->all(FLERR,"species to swap has not been selected");
  
  int nlocal = atom->nlocal;
  double **x = atom->x;
  
  for (int i = 0; i < nlocal; i++) 
    if (atom->mask[i]) // add bitgroup
      atom->type[i]= types[me%2];
  
}




/* ----------------------------------------------------------------------
   stirling approximation
   ------------------------------------------------------------------------- */
double GibbsMultiReplica::stirling(int n,int ntot)
{  
  double x1=(double)n/ntot;
  double x2=(double)(ntot-n)/ntot;
  double result=0.0;
  if(x1>0)
    result +=x1*log(x1)-x1;
  if(x2>0)
    result +=x2*log(x2)-x2;
  return result;
}



/* ----------------------------------------------------------------------
   Count the chemistry of all species
   ------------------------------------------------------------------------- */

void GibbsMultiReplica::count_chemistry()
{

  int i;
  int nlocal = atom->nlocal;
  double **x = atom->x;

  for(i=0;i<ntypes;i++)
    nchem_local[i] = 0.0;
  
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i]) {   // add bitgroup 
      int itype = atom->type[i]-1;
      nchem_local[itype]++;
    }
  }
  
  MPI_Allreduce(nchem_local,nchem,ntypes,MPI_DOUBLE,MPI_SUM,world);

}



/* ----------------------------------------------------------------------
   Pseudo-inverse of a matrix
   ------------------------------------------------------------------------- */

void GibbsMultiReplica::inverse(double A[],double x[],double B[],double fguess)
{


  // Locals 
  //  int mm = M, nn = N, nrhs = NRHS, lda = LDA, ldb = LDB, info, lwork, rank;
  int mm = nworlds, nn = nworlds, nrhs = 1, lda = nworlds, ldb = nworlds, info, lwork, rank;
  // Negative rcond means using default (machine precision) value 
  double rcond = -1.0;
  double wkopt;
  double* work;
  // Local arrays /
  // iwork dimension should be at least 3*min(m,n)*nlvl + 11*min(m,n),
  //     where nlvl = max( 0, int( log_2( min(m,n)/(smlsiz+1) ) )+1 )
  //     and smlsiz = 25 
  int iwork[3*nworlds*0+11*nworlds];
  double s[nworlds];
  double AT[nworlds*nworlds];

  // Use trasnpose due to Fortran convention
  for(int j=0;j<nworlds;j++){
    for(int i=0;i<nworlds;i++){
      //AT[i+j*nworlds] = (double)A[i*nworlds+j];
      AT[j*nworlds+i] = (double)A[j*nworlds+i];
    }
    x[j] = B[j];  // B value is overwritten with x
  }

  // Executable statements 
  //  printf( " DGELSD Example Program Results\n" )
  // Query and allocate the optimal workspace 
  lwork = -1;
  dgelsd_( &mm, &nn, &nrhs, AT, &lda, x, &ldb, s, &rcond, &rank, &wkopt, &lwork,
	   iwork, &info );
  lwork = (int)wkopt;
  work = (double*)malloc( lwork*sizeof(double) );
  // Solve the equations A*X = B 
  dgelsd_( &mm, &nn, &nrhs, AT, &lda, x, &ldb, s, &rcond, &rank, work, &lwork,
           iwork, &info );
  // Check for convergence 
  if( info > 0 ) {
    printf( "The algorithm computing SVD failed to converge;\n" );
    printf( "the least squares solution could not be computed.\n" );
    exit( 1 );
  }
  // Free workspace 
  
  //if(me_universe==0)
  //  printf( " %6f %6f\n ",B[0],B[1]);
  
  free( (void*)work );

}

