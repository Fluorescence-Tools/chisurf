import numpy as np

from .focus import focus_functions

LOW_PROBABILITY = 1.e-8
pi = 3.14159265358979


def smdif_ov6(m, d, n, q, dq, q_bg, k_rad, k_nrad, box_xy, box_z, n_ph_max, dt, t0, focus_parameter,
              focus_type, rnd_seed):
    """

    :param m: initial number of molecules of ith kind
    :param d: diffusion coef-ts
    :param n: number of detection channels
    :param q: [ij] = brightness of ith species in jth channel
    :param dq: [ij] = brightness width of ith species in jth channel, not used
    :param q_bg: background intensity
    :param k_rad: [ij]*I = i->j rate constant
    :param k_nrad: [ij] = i->j rate constant
    :param box_xy:
    :param box_z:
    :param n_ph_max: # of output photons
    :param dt: diffusion time step
    :param t0: start time
    :return:
    """
    n_species = m.shape[0]
    n_molecules = int(m.sum())                        # current number of molecules
    species = np.zeros(n_molecules, dtype=np.uint8)   # type of the molecules
    n_tw = np.zeros(n_ph_max, dtype=np.uint64)        # number of time window
    xyz = np.ones((n_molecules, 3), dtype=np.uint64)  # number of time window
    data_t = np.zeros(n_ph_max, dtype=np.uint64)      # times of photon arrival
    data_n = np.zeros(n_ph_max, dtype=np.uint8)       # channel number of photon
    data_s = np.zeros(n_ph_max, dtype=np.uint8)       # emitting species
    data_m = np.zeros(n_ph_max, dtype=np.uint32)      # emitting molecule
    focusfunction = focus_functions[focus_type]
    rmt1 = np.random.RandomState(seed=rnd_seed)
    rmt2 = np.random.RandomState(seed=rnd_seed)

    n = 0
    if t0 == 0:
        for i in range(n_species):
            if m[i] < LOW_PROBABILITY:
                continue
            t = 0.0
            while t < 1.0:
                t -= np.log(rmt1.rand() / m[i])
                while (xyz[n]**2).sum() > 1.0:
                    xyz[n, 0] = 2. * rmt1.rand() - 1.
                    xyz[n, 1] = 2. * rmt1.rand() - 1.
                    xyz[n, 2] = 2. * rmt1.rand() - 1.
                xyz[n, 0] *= box_xy
                xyz[n, 1] *= box_xy
                xyz[n, 2] *= box_z
                species[n] = i
                n += 1



"""
// diffusion of single molecules + photon statistics
// "open volume" version 2011-02-08 for c# with distributed states

#include <math.h>
#include "smdif.h"


unsigned long smdif_ov4(	/*** input ***/

 /* species definition */
 int N_species,			// number of species
 double* M,			// initial number of molecules of ith kind
 double* D,			// diffusion coef-ts
 int N_channels,		// number of detection channels
 double* q,			// [ij] = brightness of ith species in jth channel
 double* dq,		// [ij] = brightness width of ith species in jth channel, not used
 double* q_bg,			// background intensity
 double* k_rad,			// [ij]*I = i->j rate constant
 double* k_nrad,		// [ij] = i->j rate constant

 /* simulation box */
 double box_xy,
 double box_z, 			// 1/2*box size
 int focus_type,
 double* focus_param,

 double dt,			// diffusion time step
 unsigned long N_ph_max,	// # of output photons

			/*** output ***/

 unsigned long* data_T,		// number of time window
 double* data_t,		// times of photons arrivals
 short* data_N,			// number of channel for each photon
 short* data_species,		// emitting species #
 int* data_molecule,		// emitting molecule #

 unsigned long& T0,		// in: first T; out: last T
 int& N_molecules,		// current number of molecules
 double* x, double* y,
 double* z,
 short int* species,		// last state for each molecule
 double* rdq,				// random dq for each molecule

 int rmt1seed, unsigned long* rmt1state, // diffusion generator
 int& rmt1left,
 int rmt2seed, unsigned long* rmt2state, // emission generator
 int& rmt2left
)

//////////////////////////////////////////////////////////////////////////////////////
{

/* local */

const double LOW_PROBABILITY = 1.e-8;
const double pi = 3.14159265358979;
const double sqrt_2pi = sqrt(2.*pi);
const double sqrt_pi = sqrt(pi);

MTrandoms rmt1, rmt2;		// diffusion and emission random number generators
int i,j,
N_ph_buf, bg = 0;

unsigned long N_ph = 0;

double I, Iex,			// intensity at current x,y,z; I = Iex * det. eff
lambda,				// emission rate / diffusion in
k_off,				// i->(everything else) rate
tau_off,			// 1/k_off
times,				// time of emission / diffusion in
t_tr,				// time of transition
t_shift,			// time of previous transition or 0
r,				// a random number
box_xy_sq = box_xy*box_xy,
box_r_sq = box_xy_sq/box_z/box_z,
r_xy,				// x^2 + y^2
phi,				// angle in xy plane
xe, ye, ze,			// coordinates on surface
rnnorm,				// 1/norm of a normal vector at (x0,y0,z0)
step_in,			// diffusion step inside
E, Ei, f, q0,		// for distributed states
gamma=0.7853, alpha=0.017,
R, dR;

int n=0,
N_out,				// number of molecules to delete
mol_counted;			// a counter for molecules

/* seeding */
if (rmt1seed != -1) rmt1.seedMT(rmt1seed);
else rmt1.SetState(rmt1state, rmt1left);
if (rmt2seed != -1) rmt2.seedMT(rmt2seed);
else rmt2.SetState(rmt2state, rmt2left);

/* initial number of molecules and coordinates */
if (T0 == 0) {				// first call?
  if (focus_type<7) // not trace
  {
    for (i=0; i<N_species; i++) {
      if (M[i] < LOW_PROBABILITY) continue;

      times = 0.;
      while ((times -= log(rmt1.random0e1e())/M[i]) < 1.) {
        do {
          x[n] = 2.*rmt1.random0i1e()-1.;
          y[n] = 2.*rmt1.random0i1e()-1.;
          z[n] = 2.*rmt1.random0i1e()-1.;
        } while (x[n]*x[n]+y[n]*y[n]+z[n]*z[n]>1.);
        x[n] *= box_xy; y[n] *= box_xy; z[n] *= box_z;
        species[n++] = i;
      }
    }
  }
  else // trace: put round(Nfcs) molecules in the center
  {
    for (i=0; i<N_species; i++) {
      if (M[i] < LOW_PROBABILITY) continue;

      times = M[i]/4./box_xy_sq/box_z*
         focus_param[0]*focus_param[0]*focus_param[1]*3.*sqrt_pi;
      while (times-- > 0.5) {
        x[n] = 0.; y[n] = 0.; z[n] = 0.;
        x[n] *= box_xy; y[n] *= box_xy; z[n] *= box_z;
        species[n++] = i;
      }
    }
  }
  N_molecules = n;
}

/* free places */
int n_free = 0, sum_M = 0;
for (i=0; i<N_species; i++) sum_M += ceil(M[i]);

int* i_free = new int[2*sum_M+50];
n = 0; mol_counted = 0;
while (mol_counted<N_molecules) {
  if (species[n]==-1) i_free[n_free++] = n;
  else mol_counted++;
  n++;
}
for (; n<2*sum_M+50; n++) {
  i_free[n_free++] = n;
  species[n]=-1;
}


/* parameters of the ellipsoid */
double ell_f = box_z/box_xy;				// axial ratio
double ell_e = sqrt(fabs(ell_f*ell_f - 1.))/ell_f;
double ell_S_V, ell_Pzmax = 1.;				// S/V and max p(z)
if (ell_f <= 0.999999) {				// oblate
  ell_S_V = 3./2./box_z*(1.+ell_f/ell_e*log(ell_e + 1./ell_f));
  ell_Pzmax = 1./ell_f;
}
if (ell_f >= 1.000001) 					// prolate
  ell_S_V = 3./2./box_z*(1.+ell_f/ell_e*asin(ell_e));
if ((ell_f < 1.000001) && (ell_f > 0.999999)) 		// ~ sphere
  ell_S_V = 3./box_z;

/* steps, times and rates of entrance */
double* step = new double[N_species];
double flowstep_x = 0.;
double* rate_in = new double[N_species];
double* t_in = new double[N_species];
for (i=0; i<N_species; i++) {
  step[i] = sqrt(2.*D[i]*dt);
  rate_in[i] = M[i]*step[i]*ell_S_V/sqrt_2pi;
  t_in[i] = -log(rmt1.random0e1e())/rate_in[i];
}

/* focus type and parameters */
double w0 = focus_param[0];
double z0 = focus_param[1];
set_focus_parameters(focus_param);
double (*focusfunction)(double, double, double, double&);

switch (focus_type)
{
  case 0:
  focusfunction = focus_3dgauss; break;
  case 1:
  focusfunction = focus_3dgauss2; break;
  case 2:
  focusfunction = focus_rectangular; break;
  case 3:
  focusfunction = focus_cylindrical; break;
  case 4:
  focusfunction = focus_gausslorentz_pinhole; break;
  case 5:
  focusfunction = focus_gausslorentz_cyl; break;

  case 6:	// flow
  focusfunction = focus_3dgauss;
  flowstep_x = focus_param[2]*dt; break;

  case 7:	// trace
  focusfunction = focus_rectangular;
  // just in case all steps = 0
  for (i=0; i<N_species; i++) { step[i] = 0.; rate_in[i] = 0.; t_in[i] = 1.e60; }
  break;
}

/* total brightness */
double* q_total = new double[N_species];
double* dq_total = new double[N_species];
for (i=0; i<N_species; i++)  {
  q_total[i] = 0.;
  dq_total[i] = 0.;
  for (j=0; j<N_channels; j++) {
    q_total[i] += q[i*N_channels+j];
	dq_total[i] += dq[i*N_channels+j];
  }
}
double* qnow = new double[(2*sum_M+50)*N_channels];
double* qnow_total = new double[2*sum_M+50];

n = 0; mol_counted = 0;
while (mol_counted<N_molecules) {
  if (species[n]!=-1) {
    i = species[n];
	f = (q[i*N_channels]+q[i*N_channels+1])/(q[i*N_channels+2]+q[i*N_channels+3]);
	E = (1.-f*alpha)/(1+f*gamma-f*alpha);
	q0 = (q[i*N_channels]+q[i*N_channels+1])/(1.-E);

	R = 52.*pow(1./E-1.,1./6.)*(1.+0.05*rmt2.randomNorm());
	E = 1./(1.+pow(R/52.,6.));
	qnow[n*N_channels] = q0*(1.-E)*q[i*N_channels]/(q[i*N_channels]+q[i*N_channels+1]);
	qnow[n*N_channels+1] = q0*(1.-E)*q[i*N_channels+1]/(q[i*N_channels]+q[i*N_channels+1]);
	qnow[n*N_channels+2] = (q0*E*gamma+alpha*q0*(1.-E))*q[i*N_channels+2]/(q[i*N_channels+2]+q[i*N_channels+3]);
	qnow[n*N_channels+3] = (q0*E*gamma+alpha*q0*(1.-E))*q[i*N_channels+3]/(q[i*N_channels+2]+q[i*N_channels+3]);
	qnow_total[n] = qnow[n*N_channels]+qnow[n*N_channels+1]+qnow[n*N_channels+2]+qnow[n*N_channels+3];
	mol_counted++;
  }
  n++;
}

/* arrival times of bg photons */
lambda = 0.;
double* t_bg = new double[N_channels];
for (j=0; j<N_channels; j++)
  if (q_bg[j]>LOW_PROBABILITY) {
     lambda += q_bg[j];
     t_bg[j] = -log(rmt2.random0e1e())/q_bg[j];
  }
if (lambda > LOW_PROBABILITY) bg = 1;

/* buffer for storing photon events: */

 /* estimating max. possible number of photons per time_window */
 for (i=0; i<N_species; i++)
  for (j=0; j<N_channels; j++)  lambda += 4.*q[i*N_channels+j]*M[i];

 int buf_size = 2*ceil(lambda*dt) + 50; 	// roughly, p(overflow)<1e-15
 double* buf_t = new double[buf_size];
 int* buf_N = new int[buf_size];
 int* buf_species = new int[buf_size];
 int* buf_molecule = new int[buf_size];

/* transitions */
double* k_off_rad = new double[N_species];
double* k_off_nrad = new double[N_species];
for (i=0; i<N_species; i++) {
  k_off_rad[i] = 0.;
  k_off_nrad[i] = 0.;
  for (j=0; j<N_species; j++) {
	k_off_rad[i] += k_rad[i*N_species+j];
	k_off_nrad[i] += k_nrad[i*N_species+j];
  }
}

//////////////////////////////////////////////////////////////////////////////////////

while (N_ph<N_ph_max) {

  N_ph_buf = 0;
  n = 0; mol_counted = 0; N_out = 0;

  while (mol_counted<N_molecules) {

    i = species[n];
    if (i == -1) { n++; continue; }

    /* intensity at (x,y,z) */
    I = focusfunction(x[n], y[n], z[n], Iex);

    t_tr = 0.;
    do {

      t_shift = t_tr;

      /* time of the next transition */

      k_off = Iex*k_off_rad[i] + k_off_nrad[i];
      tau_off = 1./k_off;
      if (k_off > LOW_PROBABILITY)
        t_tr = t_shift-log(rmt1.random0e1e())*tau_off;
      else t_tr = dt+1.;

      /*** emission ***/

      /* average (expected) number of photons per dt=1 */

      lambda = I*(qnow_total[n]);
      if (lambda > LOW_PROBABILITY) {

        /* arrival time of the first photon */
        times = t_shift-log(rmt2.random0e1e())/lambda;

        /* more photons? */
        while ((times<t_tr) && (times<dt)) {
          /* determine number of channel */
          r = 1. - rmt2.random0i1e();
          j = -1;
          while (r>0.) {
            j++;
            r -= qnow[n*N_channels+j]/qnow_total[n];
          }
          buf_N[N_ph_buf] = j;

          buf_t[N_ph_buf] = times;
          buf_species[N_ph_buf] = i;
          buf_molecule[N_ph_buf++] = n;
          times -= log(rmt2.random0e1e())/lambda; /* arrival time of the next photon */
        }
      }

      /* new state */
      if (t_tr<dt) {
        r = 1.-rmt1.random0i1e();
        j = -1;
        while (r>0.) {
          j++;
          r -= tau_off*(Iex*k_rad[i*N_species+j] + k_nrad[i*N_species+j]);
        }
        i=j;
	    f = (q[i*N_channels]+q[i*N_channels+1])/(q[i*N_channels+2]+q[i*N_channels+3]);
	    E = (1.-f*alpha)/(1+f*gamma-f*alpha);
	    q0 = (q[i*N_channels]+q[i*N_channels+1])/(1.-E);
	    R = 52.*pow(1./E-1.,1./6.)*(1.+0.05*rmt2.randomNorm());
	    E = 1./(1.+pow(R/52.,6.));
	    qnow[n*N_channels] = q0*(1.-E)*q[i*N_channels]/(q[i*N_channels]+q[i*N_channels+1]);
	    qnow[n*N_channels+1] = q0*(1.-E)*q[i*N_channels+1]/(q[i*N_channels]+q[i*N_channels+1]);
	    qnow[n*N_channels+2] = (q0*E*gamma+alpha*q0*(1.-E))*q[i*N_channels+2]/(q[i*N_channels+2]+q[i*N_channels+3]);
	    qnow[n*N_channels+3] = (q0*E*gamma+alpha*q0*(1.-E))*q[i*N_channels+3]/(q[i*N_channels+2]+q[i*N_channels+3]);
	    qnow_total[n] = qnow[n*N_channels]+qnow[n*N_channels+1]+qnow[n*N_channels+2]+qnow[n*N_channels+3];

      }

    } while (t_tr<dt);

    species[n]=i;

    /* diffusion step */
    x[n] += step[i]*rmt1.randomNorm() + flowstep_x ;
    y[n] += step[i]*rmt1.randomNorm();
    z[n] += step[i]*rmt1.randomNorm();

    /* out? -- free space */
    if (x[n]*x[n]+y[n]*y[n]+box_r_sq*z[n]*z[n]>box_xy_sq) {
      i_free[n_free++] = n;
      species[n] = -1;
      N_out++;
    }

    mol_counted++; n++;			// next molecule
  }
  N_molecules -= N_out;

  /* adding background */
  if (bg) {
    for (j=0; j<N_channels; j++)  {
      if (q_bg[j] < LOW_PROBABILITY) continue;
      while (t_bg[j]<dt) {
        buf_t[N_ph_buf] = t_bg[j];
        buf_N[N_ph_buf] = j;
        buf_species[N_ph_buf] = N_species;
        buf_molecule[N_ph_buf++] = 0;
        t_bg[j] -= log(rmt2.random0e1e())/q_bg[j]; /* arrival time of the next photon */
      }
      t_bg[j] -= dt;
    }
  }

  /* sorting buffer according to times and copying it to the data */
  if (N_ph_buf >= 2) shell4(N_ph_buf, buf_t, buf_N, buf_species, buf_molecule);

  for (j=0; j<N_ph_buf; j++) {
    data_T[N_ph] = T0;
    data_t[N_ph] = buf_t[j];
    data_N[N_ph] = buf_N[j];
    data_species[N_ph] = buf_species[j];
    data_molecule[N_ph++] = buf_molecule[j];
  }

  /* sorting ifree */
  if ((n_free >= 2) && (N_out > 0)) shell_r(n_free,i_free);

  /* new molecules */
  for (i=0; i<N_species; i++) {

    /* average (expected) number of new molecules */
    if (rate_in[i] < LOW_PROBABILITY) continue;

    while (t_in[i] <= 1.) {	/* poissonian */

      /* adding molecules */
      n = i_free[--n_free];

      /* seed on surface -- from the EFT project */
      do {
        ze = 2.*rmt1.random0i1e() - 1.;
        r = rmt1.random0i1e()*ell_Pzmax;
      } while (r*r > 1. - ze*ze*(1. - box_r_sq)); // reject z

      r_xy = sqrt(1.-ze*ze) * box_xy;
      ze *= box_z;

      phi = rmt1.random0i1e()*2.*pi;
      xe = cos(phi)*r_xy;
      ye = sin(phi)*r_xy;

      /* moving inside */
      step_in = step[i]*random_erfc(rmt1);
      rnnorm = 1./sqrt(r_xy*r_xy + ze*ze*box_r_sq*box_r_sq);
      x[n] = xe * (1.-step_in*rnnorm);
      y[n] = ye * (1.-step_in*rnnorm);
      z[n] = ze * (1.-step_in*rnnorm*box_r_sq);

      species[n] = i;
	  // dq now has a meaning of sigma_R
	  f = (q[i*N_channels]+q[i*N_channels+1])/(q[i*N_channels+2]+q[i*N_channels+3]);
	  E = (1.-f*alpha)/(1+f*gamma-f*alpha);
	  q0 = (q[i*N_channels]+q[i*N_channels+1])/(1.-E);
	  R = 52.*pow(1./E-1.,1./6.)*(1.+0.05*rmt2.randomNorm());
	  E = 1./(1.+pow(R/52.,6.));
	  qnow[n*N_channels] = q0*(1.-E)*q[i*N_channels]/(q[i*N_channels]+q[i*N_channels+1]);
	  qnow[n*N_channels+1] = q0*(1.-E)*q[i*N_channels+1]/(q[i*N_channels]+q[i*N_channels+1]);
	  qnow[n*N_channels+2] = (q0*E*gamma+alpha*q0*(1.-E))*q[i*N_channels+2]/(q[i*N_channels+2]+q[i*N_channels+3]);
	  qnow[n*N_channels+3] = (q0*E*gamma+alpha*q0*(1.-E))*q[i*N_channels+3]/(q[i*N_channels+2]+q[i*N_channels+3]);
	  qnow_total[n] = qnow[n*N_channels]+qnow[n*N_channels+1]+qnow[n*N_channels+2]+qnow[n*N_channels+3];

	  //////////////////////////////////
      N_molecules++;
      t_in[i] -= log(rmt1.random0e1e())/rate_in[i];
    }
    t_in[i] -= 1.;
  }

  T0++;				// next time window
}

//////////////////////////////////////////////////////////////////////////////////////

delete[] i_free; delete[] k_off_rad; delete[] k_off_nrad;
delete[] buf_t; delete[] buf_N; delete[] buf_species; delete[] buf_molecule;
delete[] rate_in; delete[] t_in; delete[] q_total; delete[] t_bg; delete[] step; delete[] dq_total;

rmt1.GetState(rmt1state, rmt1left);
rmt2.GetState(rmt2state, rmt2left);

return N_ph;

}

// everything -> burbulator.dll
// f /TP /O2 /EHsc /MD /Feburbulator.dll data2spc_tac.cpp smdif_ov4.cpp smdif_ov3.cpp smdif_misc.cpp focus.cpp rotdiff.cpp mt19937cok.cpp /link /dll /def:burbulator.def
// VC8: macro_times /manifest burbulator.dll.manifest /outputresource:"burbulator.dll;#2"

// everything -> burbulator_x64.dll
// f /TP /O2 /EHsc /MD /Feburbulator_x64.dll data2spc_tac.cpp smdif_ov4.cpp smdif_ov3.cpp smdif_misc.cpp focus.cpp rotdiff.cpp mt19937cok.cpp /link /dll /def:burbulator_x64.def
// VC8: macro_times /manifest burbulator_x64.dll.manifest /outputresource:"burbulator_x64.dll;#2"
"""