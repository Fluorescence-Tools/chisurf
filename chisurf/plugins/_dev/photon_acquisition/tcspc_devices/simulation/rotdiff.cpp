// rotational (only) diffusion of single molecules + photon statistics
// version 2007-08-09 
// probability of having several photons per time step neglected

#include <math.h>
#include "mtrandom.h"

/* normalization, exact and approximate (sqrt up to 3rd order) */
void inline normalize(double& x0, double& y0, double& z0)
{

	double rxyz = 1./sqrt(x0*x0 + y0*y0 + z0*z0);

	x0 *= rxyz;
	y0 *= rxyz;
	z0 *= rxyz;
}
void inline normalize_4o(double& x0, double& y0, double& z0)
{

	double rxyz = x0*x0 + y0*y0 + z0*z0 - 1.;
	rxyz = 1. + rxyz*(-0.5 + rxyz*(0.375 - 0.3125*rxyz));

	x0 *= rxyz;
	y0 *= rxyz;
	z0 *= rxyz;
}

//////////////////////////////////////////////////////////////////////////////////////

unsigned long rotdiff(	/*** input ***/

	double D,			// rotational diffusion coef-t
	double tau,			// fl lifetime
	double r0,			// limiting anisotropy
	double l1, double l2,		// "Japanese" factors
	double q,			// brightness

	double dt,			// diffusion time step
	unsigned long N_ph_max,		// # of output photons

	/*** output: compatible with data2spc ***/

	unsigned long* data_T,		// number of time window
	double* data_t,			// times of photons arrivals
	short* data_N,			// number of channel for each photon
	short* data_species,		// emitting species #
	int* data_molecule,		// emitting molecule #

	unsigned long& T0,		// in: first T; out: last T
	double& x, double& y,		// last state (ex transition)
	double& z,

	int rmt1seed, unsigned long* rmt1state, // diffusion generator
	int& rmt1left,
	int rmt2seed, unsigned long* rmt2state, // emission generator
	int& rmt2left
)

//////////////////////////////////////////////////////////////////////////////////////
{

	/* local */

	const double pi = 3.14159265358979;

	MTrandoms rmt1, rmt2;		// diffusion and emission random number generators
	int j;

	unsigned long N_ph = 0;

	double	t = -1.,		// time of emission
		r,			// a random number
		phi,			// angle in xy plane
		pp, ps,			// parallel and perp. em probabilities
		xn1, yn1, zn1,
		xn2, yn2, zn2,		// two normal vectors to (x,y,z)
		xem, yem, zem;		// emission transition dipole

	/* seeding */
	if (rmt1seed != -1) rmt1.seedMT(rmt1seed);
	else rmt1.SetState(rmt1state, rmt1left);
	if (rmt2seed != -1) rmt2.seedMT(rmt2seed);
	else rmt2.SetState(rmt2state, rmt2left);

	/* initial number of molecules and coordinates */
	if (T0 == 0) {				// first call?
		z = -1. + rmt1.random0i1e()*2.;
		phi = rmt1.random0i1e()*2.*pi;
		x = cos(phi)*sqrt(1.-z*z);
		y = sin(phi)*sqrt(1.-z*z);
	}

	double step = sqrt(2.*D*dt);

	/* r0 */
	double th0 = acos(sqrt((5.*r0+1.)/3.));
	double tg_th0 = tan(th0);
	double r0dir, cos_r0dir, sin_r0dir;

	/* l1, l2 */
	double l1l2f;
	l1 > l2 ? l1l2f = 1./(1. - l2 + l1) : l1l2f = 1./(1. - l1 + l2);

//////////////////////////////////////////////////////////////////////////////////////

	while (N_ph<N_ph_max) {

		j = -1;

		if (t < 0)				// excitation possible?
		{
			if (rmt2.random0i1e() < 3.*x*x*q*dt)
				t = rmt2.random0e1e()*dt-log(rmt2.random0e1e())*tau;
		}

		// emission
		if ((t>=0.) && (t<dt))						 
		{
			// modelling r0 effect: rotate by theta
			/* two normal vectors */
			if (fabs(z) < 0.9) { xn1 = y; yn1 = -x; zn1 = 0.; }
			else { xn1 = 0.; yn1 = z; zn1 = -y; }
			normalize(xn1, yn1, zn1);
			xn2 = y*zn1 - z*yn1; yn2 = z*xn1 - x*zn1; zn2 = x*yn1 - y*xn1;
			r0dir = rmt1.random0i1e()*2.*pi;
			cos_r0dir = cos(r0dir); sin_r0dir = sin(r0dir); 
			xem = x + tg_th0*(cos_r0dir*xn1 + sin_r0dir*xn2);
			yem = y + tg_th0*(cos_r0dir*yn1 + sin_r0dir*yn2);
			zem = z + tg_th0*(cos_r0dir*zn1 + sin_r0dir*zn2);
			normalize(xem, yem, zem);

			// modelling emission + l1 and l2 effects
			r = rmt2.random0e1e();
			pp = l1l2f*((1.-l1)*xem*xem + l1*yem*yem);	// parallel channel
			ps = l1l2f*(l2*xem*xem + (1.-l2)*yem*yem);	// perpendicular channel
			if (r < pp) j = 0;
			else if (r < pp + ps) j = 1;
		}

		if (j>=0)	// add a photon
		{
			data_T[N_ph] = T0;
			data_N[N_ph] = j;

			data_t[N_ph] = t;
			data_species[N_ph] = 0;
			data_molecule[N_ph++] = 0;
		}

		/* rotational diffusion step */
		x += step*rmt1.randomNorm();
		y += step*rmt1.randomNorm();
		z += step*rmt1.randomNorm();     
		normalize_4o(x,y,z);

		t -= dt;
		T0++;				// next time window
	}



//////////////////////////////////////////////////////////////////////////////////////

rmt1.GetState(rmt1state, rmt1left);
rmt2.GetState(rmt2state, rmt2left);

return N_ph;

}

// cl /TP /O2 /EHsc /MD /Ferotdiff.dll rotdiff.cpp mt19937cok.cpp /link /dll /export:rotdiff
// VC8: mt /manifest rotdiff.dll.manifest /outputresource:"rotdiff.dll;#2"
