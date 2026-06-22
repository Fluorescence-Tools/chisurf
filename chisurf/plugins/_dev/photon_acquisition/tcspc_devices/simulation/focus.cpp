// laser focus geometry

#include<math.h>
#include<stdlib.h>

const double sqrt_8 = sqrt(8.);
const double pi = 3.14159265358979;

// global focus parameters
static double w0;
static double z0;
static double res0;	// lens resolution
static double tgalpha;	// tan of the aperture half-angle
static double Rph;	// pinhole radius 
static double z0_CEF;	// cutoff for cylindrical CEF

void set_focus_parameters(double* focus_param)
{
 w0 = focus_param[0];
 z0 = focus_param[1];
 res0 = focus_param[2];
 tgalpha = focus_param[3];
 Rph = focus_param[4];
 z0_CEF = focus_param[5];
}

// 3D Gaussian, uniform CEF = 1
double focus_3dgauss(double x, double y, double z, double& Iex)
{
 Iex = sqrt_8*exp(-2.*((x*x+y*y)/(w0*w0)+z*z/(z0*z0)));
 return Iex;
}

// 3D Gaussian excitation and CEF
double focus_3dgauss2(double x, double y, double z, double& Iex)
{
 Iex = sqrt_8*exp(-((x*x+y*y)/(w0*w0)+z*z/(z0*z0)));
 return Iex*Iex/sqrt_8;
}

// rectangular excitation, uniform CEF
double focus_rectangular(double x, double y, double z, double& Iex)
{
 Iex = (abs(x)<w0) && (abs(y)<w0) && (abs(z)<z0);
 return Iex;
}

// cylindrical excitation, uniform CEF
double focus_cylindrical(double x, double y, double z, double& Iex)
{
 Iex = (x*x+y*y<w0*w0) && (abs(z)<z0);
 return Iex;
}

// Gaussian-Lorentzian excitation, pinhole CEF
double focus_gausslorentz_pinhole(double x, double y, double z, double& Iex)
{
 // gaussian-lorentzian excitation
 double rz = (1. + z*z/(z0*z0));
 double d = sqrt(x*x + y*y);
 Iex = sqrt_8*exp(-2.*d*d/(w0*w0*rz))/rz;

 // CEF: Rigler et al, Eur. Biophys. J. 22 (1993) 169 
 double Iem, r, S_psf, a, b, cosa, cosb;

 r = sqrt(res0*res0 + z*z*tgalpha*tgalpha);
 S_psf = pi*r*r;

 if (d<=abs(Rph-r))
   Iem = __min(1., Rph*Rph/(r*r));
 else if (d>=Rph+r)
   Iem = 0.;
 else {
   cosa = (Rph*Rph-r*r+d*d)/2./d/Rph;
   cosb = (r*r-Rph*Rph+d*d)/2./d/r;
   a = acos(cosa);
   b = acos(cosb);
   Iem = 1./S_psf*(Rph*Rph*(a-cosa*sin(a))+r*r*(b-cosb*sin(b)));
 }

 return Iem*Iex;
}

// Gaussian-Lorentzian excitation, cylindrical CEF
double focus_gausslorentz_cyl(double x, double y, double z, double& Iex)
{
 // gaussian-lorentzian excitation
 double rz = (1. + z*z/(z0*z0));
 double d = sqrt(x*x + y*y);
 Iex = sqrt_8*exp(-2.*d*d/(w0*w0*rz))/rz;

 return Iex * ((x*x+y*y<Rph*Rph) && (abs(z)<z0_CEF));
}

