#include "mtrandom.h"

//////////////////////////////////// smdif_misc.cpp ///////////////////////////////////

void shell2(unsigned long, double*, int*);
void shell4(unsigned long, double*, int*, int*, int*);
void shell_r(unsigned long, int*);
double qnorm(double);
double random_erfc(MTrandoms&);

////////////////////////////////////// focus.cpp //////////////////////////////////////

void set_focus_parameters(double*);
double focus_3dgauss(double, double, double, double&);
double focus_3dgauss2(double, double, double, double&);
double focus_rectangular(double, double, double, double&);
double focus_cylindrical(double, double, double, double&);
double focus_gausslorentz_pinhole(double, double, double, double&);
double focus_gausslorentz_cyl(double, double, double, double&);

///////////////////////////////////// smdif_ov3.cpp ///////////////////////////////////

unsigned long smdif_ov3(int, double*, double*, int, double*, double*, double*, double*, 
	double, double, int, double*,
	double, unsigned long,	
	unsigned long*,	double*, short*, short*, int*,		
	unsigned long&,	int&, double*, double*, double*, short*,
        int, unsigned long*, int&,
	int, unsigned long*, int&);

///////////////////////////////////// rotdiff.cpp /////////////////////////////////////

unsigned long rotdiff(
	double, double, double, double, double, double,
	double, unsigned long,
	unsigned long*, double*, short*, short*, int*,
	unsigned long&,
	double&, double&, double&,
	int, unsigned long*, int&,
	int, unsigned long*, int&);

///////////////////////////////////// data2spc_tac.cpp ////////////////////////////////

int data2spc132_tac(
	int, int,
	unsigned long*, double*, short*, short*, int*,
	unsigned long, double,
	unsigned short*,
	int, double, double,
	double*, int*,
	char*,
	unsigned long&, unsigned long&,
	unsigned long*, int&);