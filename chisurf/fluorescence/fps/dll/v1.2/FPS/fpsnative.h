// fpsnative.h
#include <intrin.h>


//////// AV ///////////


int calculate3R(double L, double W, double R1, double R2, double R3, int atom_i, double dg,
	double* XLocal, double* YLocal, double* ZLocal,
	double* vdWR, int NAtoms, double vdWRMax,
	double linkersphere, int linknodes,
	unsigned char* density);


//////// misc /////////////

int testnative();
