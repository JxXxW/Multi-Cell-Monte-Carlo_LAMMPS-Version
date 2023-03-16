/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMMAND_CLASS

CommandStyle(gibbs/multireplica,GibbsMultiReplica)

#else

#ifndef LMP_GIBBS_MULTIREPLICA_H
#define LMP_GIBBS_MULTIREPLICA_H

#include "pointers.h"

namespace LAMMPS_NS {

class GibbsMultiReplica : protected Pointers {
 public:
  GibbsMultiReplica(class LAMMPS *);
  ~GibbsMultiReplica();
  void command(int, char **);
  int remapflag;                   // whether x,v are remapped across PBC
  int aveflag;                     // whether thermalization occurs between mc moves
  int dimflag[6];                  // which dims are barostatted

 private:
  int me,me_universe;          // my proc ID in world and universe
  int iworld,nworlds,nprocs;   // world info
  int ntypes;                  // number of types on system
  double boltz;                // copy from output->boltz
  MPI_Comm roots;              // MPI comm with 1 root proc from each world
  MPI_Comm intra;              // MPI comm with processor within each world
  class RanMars *ranrepl;
  class RanPark *random_equal,*ranboltz;  // RNGs for swapping and Boltz factor
  class RanPark *ranworld;     // RNGs for selecting particles in each world
  class FixStore *fix_revert;        // revert state                                                                                                 
  int nevery;                  // # of timesteps between moves
  int nmoves;                  // # of move attempts to perform
  int ncycles;                 // # thermalization cycles 
  int seed_swap,seed_boltz;    // seed initializing random numbers
  int whichfix;                // index of temperature fix to use
  int fixstyle;                // what kind of temperature fix is used
  int my_set_temp;             // which set temp I am simulating
  double *set_temp;            // static list of multireplica set temperatures
  int *root2world;             // 
  int *world2root;             // world2root[i] = root proc of world i
  double *world_chemistry;        // world2temp[i] = chemistry of types[0] in world i
  int swap_inter,swap_intra,flip_inter,vol_inter,exch_inter;
  int swap_inter_all,swap_intra_all,flip_inter_all,vol_inter_all,exch_inter_all;
  int sintra_suc,sinter_suc,flip_suc,exch_suc,vol_suc;
  int sintra_att,sinter_att,flip_att,exch_att,vol_att;
  int typesflag,mcflag,slopeflag; 
  int groupbitall;          // group bitmask for inserted atoms                                                                       

  double xlo,xhi,ylo,yhi,zlo,zhi;
  double region_xlo,region_xhi,region_ylo,region_yhi,region_zlo,region_zhi;
  double region_volume;
  double *sublo,*subhi;

  
  double time_comm,time_output,time_start;  // timer variables

  double Ntot;                  // total particles in whole universe
  double *nchem,*nchem_local;
  int ftypes;                   // # atoms species to flip
  int nxchg;                   // # of atoms on all procs                                                                     
  int nxchg_local;             // # of atoms on this proc                                                                     
  int nxchg_before;            // # of atoms on procs < this proc 

  int niswap,njswap;               // # of i and j swap atoms on this world
  int niswap_local,njswap_local;   // # of swap atoms on this proc                                                                  
  int niswap_before,njswap_before; // # of swap atoms on procs < this proc 

  int flip_nmax;  // 
  double energy_stored,save_ene,wtmp,werr,werr_partner,molene,fudge,vol_stored,dens_stored;
  double dmu,*mupair,*muavg,*enerun,*predrun;
  double *facc,*fatt;
  double p_hydro,vol0;

  int atom_all_nmax,atom_i_intra_nmax,atom_j_intra_nmax,atom_many_nmax;
  int *type_list,*type2list;
  int *local_all_atoms_list;
  int *local_swap_iatom_list;
  int *local_swap_jatom_list;
  int *local_katoms_list;
  double *sqrt_mass_ratio;
  double *init_conc,*conc_matrix,*B_vect;
  double *molar_fraction,*wmolfrac;         // molar_fraction[i] = molar fraction of types[0] in world i
  double *emol,*mu,*mu_vect,*mu_matrix;        
  double *farray,*barray;     // arrays containing estimate of chemical potential and virtual attempts 
  double *carray,*darray,*marray;
  double **runavg;


  struct Set {
    double p_start,p_stop,p_target;
    double lo_target,hi_target;
    double tilt_target,tilt_flip;
    double amplitude,delta;
    int style,substyle;
  };
  Set *set;
  int dimension;
  int press_couple,allremap;


 
  char *min_style;

  void print_status();
  void dynamics();
  void minimization();
  int attempt_volume_moves(int [],int,double);
  int attempt_exchange(int [],int,double);
  int attempt_atomic_insertion(int []);
  int attempt_atomic_deletion(int i);
  int attempt_atomic_translation_full(double);
  int attempt_inter_swap(int [],int,double,int []);
  int attempt_intra_swap(int [],double);
  int attempt_flip(int , int [],int ,double,double);
  int attempt_global_flip(int [] , int [],int ,double,double);
  //  int attempt_many_flips(int [], int [],int ,double);
  void store_state();
  void revert_state();
  int pick_random_atom();
  int pick_i_swap_atom();
  int pick_j_swap_atom();
  void update_all_atoms_list();
  void update_swap_atoms_list(int arr[]);
  double energy_full(int);
  double compute_vector(int );
  void compute_press_target();
  void count_chemistry();
  double molar_free_energy(int,int);
  double stirling(int,int);
  void set_world_chemistry(int []);
  void Widom_test(double,int,int,int);
  int check_molfrac(double f[]);
  void inverse(double [],double [],double [],double);
  void pseudo_inverse(double [],double [],double [],double);

  
  class Compute *c_pe,*c_press,*c_temp;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Must have more than one processor partition to temper

Cannot use the temper command with only one processor partition.  Use
the -partition command-line option.

E: GibbsMultireplica command before simulation box is defined

The temper command cannot be used before a read_data, read_restart, or
create_box command.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: GibbsMultireplicaing fix ID is not defined

The fix ID specified by the temper command does not exist.

E: Illegal temperature index

UNDOCUMENTED

E: Invalid frequency in temper command

Nevery must be > 0.

E: Non integer # of swaps in temper command

Swap frequency in temper command must evenly divide the total # of
timesteps.

E: GibbsMultireplica temperature fix is not supported

UNDOCUMENTED

E: Too many timesteps

The cumulative timesteps must fit in a 64-bit integer.

E: GibbsMultireplicaing could not find thermo_pe compute

This compute is created by the thermo command.  It must have been
explicitly deleted by a uncompute command.

U: GibbsMultireplica temperature fix is not valid

The fix specified by the temper command is not one that controls
temperature (nvt or langevin).

*/
