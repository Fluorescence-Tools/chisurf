// Calculating p(G,R) = SgSr for Tatiana, all functions
// version 2006-09-05 for LabView

// VC7: cl /O2 /EHsc /MD /Fesgsr.dll sgsr.cpp sgsr_misc.cpp /link /dll /DEF:sgsr.def
// VC8+ mt /manifest sgsr.dll.manifest /outputresource:"sgsr.dll;#2"

#include "sgsr.h"
#include "stdio.h"

////////////////////////////////// convolution with background ////////////////////////////////////

void conv_pF (double* SgSr, double* FgFr,
                     unsigned int Nmax, 
                     double Bg, double Br)
{

 double* tmp = new double [(Nmax+1)*(Nmax+1)];

 // green and red background 
 double* bg = new double [Nmax+1];
 poisson_0toN(bg, Bg, Nmax);
 unsigned int Bg_max = 2*(unsigned int)Bg+52;
 double* br = new double [Nmax+1];
 poisson_0toN(br, Br, Nmax);
 unsigned int Br_max = 2*(unsigned int)Br+52;
 
 // sum
 double s;
 unsigned int j, i, i_start, red, green;

 for (red = 0; red<=Nmax; red++) {
   red>Br_max ? i_start = red-Br_max : i_start = 0;
   for (green = 0; green<=Nmax-red; green++) {
      s = 0.;
      j = (Nmax+1)*green;
      for (i=i_start; i<=red; i++) s += FgFr[j+i] * br[red-i];
      tmp[(Nmax+1)*red+green] = s;
    }
 }  
 for (green = 0; green<=Nmax; green++) {
   green>Bg_max ? i_start = green-Bg_max : i_start = 0;
   for (red = 0; red<=Nmax-green; red++) {
      s = 0.;
      j = (Nmax+1)*red;
      for (i=i_start; i<=green; i++) s += tmp[j+i] * bg[green-i];
      SgSr[(Nmax+1)*green+red] = s;
    }
 }

 delete[] bg; delete[] br; delete[] tmp;
}

/***** modification for P(N) *****/

static void conv_pN (double* SgSr, double* FgFr, 
                     unsigned int Nmax, 
                     double Bg, double Br,
                     double* pN)
{


 double s;
 unsigned int j, i;

 conv_pF(SgSr, FgFr, Nmax, Bg, Br);

 for (i = 0; i<=Nmax; i++) {
   s = 0.;
   for (j = 0; j<=i; j++) s += SgSr[Nmax*j+i];
   for (j = 0; j<=i; j++) SgSr[Nmax*j+i] *= pN[i]/s;
 }

}


/////////////////////////// calculating p(G,R) according to Matthew ///////////////////////////////

void sgsr_pN (double* SgSr, 		// return: SgSr(i,j) = p(Sg=i, Sr=j)
	      double* pN,		// p(N)
	      unsigned int Nmax, 	// max number of photons (max N)
	      double Bg,		// <background green>, per time window (!)
	      double Br,		// <background red>, -"-
	      double pg_theor)		// p(green|F=1), incl. crosstalk
{

 /*** FgFr: matrix, FgFr(i,j) = p(Fg=i, Fr=j | F=i+j) ***/

 double* FgFr = new double [(Nmax+1)*(Nmax+1)]; FgFr[0] = 1.;
 unsigned int i,red;

 for (i = 1; i<=Nmax; i++) {
     polynom2_conv (FgFr+i*(Nmax+1), FgFr+(i-1)*(Nmax+1), i, pg_theor);
     for (red = 0; red<=i-1; red++) 
       FgFr[(i-1-red)*(Nmax+1)+red] = FgFr[(i-1)*(Nmax+1)+red];
  }
 for (red = 0; red<=Nmax; red++) 
   FgFr[(Nmax-red)*(Nmax+1)+red] = FgFr[Nmax*(Nmax+1)+red];


 /*** SgSr: matrix, SgSr(i,j) = p(Sg = i, Sr = j) ***/

 conv_pN(SgSr, FgFr, Nmax, Bg, Br, pN);

 delete[] FgFr;
}


///////////////////////// calculating p(G,R), one ratio, one P(F) ///////////////////////////////
void sgsr_pF (double* SgSr,		// see sgsr_pN
              double* pF,		// input: p(F)
              unsigned int Nmax,
	      double Bg,
              double Br,
              double pg_theor)
{

 /*** FgFr: matrix, FgFr(i,j) = p(Fg = i, Fr = j) ***/

 double* FgFr = new double [(Nmax+1)*(Nmax+1)]; FgFr[0] = 1.;
 unsigned int i,red;

 for (i = 1; i<=Nmax; i++) {
     polynom2_conv (FgFr+i*(Nmax+1), FgFr+(i-1)*(Nmax+1), i, pg_theor);
     for (red = 0; red<=i-1; red++)
       FgFr[(i-1-red)*(Nmax+1)+red] = FgFr[(i-1)*(Nmax+1)+red]*pF[i-1];
  }
 for (red = 0; red<=Nmax; red++)
   FgFr[(Nmax-red)*(Nmax+1)+red] = FgFr[Nmax*(Nmax+1)+red]*pF[Nmax];


 /*** SgSr: matrix, SgSr(i,j) = p(Sg = i, Sr = j) ***/

 conv_pF(SgSr, FgFr, Nmax, Bg, Br);

 delete[] FgFr;
}


///////////////////////// calculating p(G,R), several ratios, same P(N) ///////////////////////////

void sgsr_pN_manypg (double* SgSr,		// see sgsr_pN
		     double* pN,		// input: p(N)
		     unsigned int Nmax,
		     double Bg,
		     double Br,
		     unsigned int N_pg,		// size of pg_theor
		     double* pg_theor,
		     double* a)			// corresponding amplitudes
{


 /*** FgFr: matrix, FgFr(i,j) = p(Fg = i, Fr = j) ***/

 double* FgFr = new double [(Nmax+1)*(Nmax+1)];
 double* tmp = new double [(Nmax+1)*(Nmax+1)]; 
 unsigned int j, i, red;

 for (j = 0; j<(Nmax+1)*(Nmax+1); j++) FgFr[j] = 0.;

 for (j = 0; j<N_pg; j++) {

   tmp[0] = 1.;
   for (i = 1; i<=Nmax; i++) {
     polynom2_conv (tmp+i*(Nmax+1), tmp+(i-1)*(Nmax+1), i, pg_theor[j]);
     for (red = 0; red<=i-1; red++) 
       FgFr[(i-1-red)*(Nmax+1)+red] += tmp[(i-1)*(Nmax+1)+red]*a[j];
    }
   for (red = 0; red<=Nmax; red++) 
     FgFr[(Nmax-red)*(Nmax+1)+red] += tmp[Nmax*(Nmax+1)+red]*a[j];
 }

 /*** SgSr: matrix, SgSr(i,j) = p(Sg = i, Sr = j) ***/

 conv_pN(SgSr, FgFr, Nmax, Bg, Br, pN);

 delete[] FgFr; delete[] tmp;

}

///////////////////////// calculating p(G,R), several ratios, same P(F) ///////////////////////////

void sgsr_pF_manypg (double* SgSr,		// see sgsr_pN
		     double* pF,		// input: p(F)
		     unsigned int Nmax,
		     double Bg,
		     double Br,
		     unsigned int N_pg,		// size of pg_theor
		     double* pg_theor,
		     double* a)			// corresponding amplitudes
{


 /*** FgFr: matrix, FgFr(i,j) = p(Fg = i, Fr = j) ***/

 double* FgFr = new double [(Nmax+1)*(Nmax+1)];
 double* tmp = new double [(Nmax+1)*(Nmax+1)]; 
 unsigned int j, i, red;

 for (j = 0; j<(Nmax+1)*(Nmax+1); j++) FgFr[j] = 0.;

 for (j = 0; j<N_pg; j++) {

   tmp[0] = 1.;
   for (i = 1; i<=Nmax; i++) {
     polynom2_conv (tmp+i*(Nmax+1), tmp+(i-1)*(Nmax+1), i, pg_theor[j]);
     for (red = 0; red<=i-1; red++) 
       FgFr[(i-1-red)*(Nmax+1)+red] += tmp[(i-1)*(Nmax+1)+red]*pF[i-1]*a[j];
    }
   for (red = 0; red<=Nmax; red++) 
     FgFr[(Nmax-red)*(Nmax+1)+red] += tmp[Nmax*(Nmax+1)+red]*pF[Nmax]*a[j];
 }

 /*** SgSr: matrix, SgSr(i,j) = p(Sg = i, Sr = j) ***/

 conv_pF(SgSr, FgFr, Nmax, Bg, Br);

 delete[] FgFr; delete[] tmp;

}

///////////////////////// calculating p(G,R), several ratios, many P(F)s //////////////////////////

void sgsr_manypF    (double* SgSr,		// see sgsr_pN
		     double* pF,		// input: p(F); size = (Nmax+1)xN_pg
		     unsigned int Nmax,
		     double Bg,
		     double Br,
		     unsigned int N_pg,		// size of pg_theor
		     double* pg_theor,
		     double* a)			// corresponding amplitudes
{


 /*** FgFr: matrix, FgFr(i,j) = p(Fg = i, Fr = j) ***/

 double* FgFr = new double [(Nmax+1)*(Nmax+1)];
 double* tmp = new double [(Nmax+1)*(Nmax+1)]; 
 unsigned int j, i, red;

 for (j = 0; j<(Nmax+1)*(Nmax+1); j++) FgFr[j] = 0.;

 for (j = 0; j<N_pg; j++) {

   tmp[0] = 1.;
   for (i = 1; i<=Nmax; i++) {
     polynom2_conv (tmp+i*(Nmax+1), tmp+(i-1)*(Nmax+1), i, pg_theor[j]);
     for (red = 0; red<=i-1; red++)
       FgFr[(i-1-red)*(Nmax+1)+red] += tmp[(i-1)*(Nmax+1)+red]*pF[(Nmax+1)*j+i-1]*a[j];
    }
   for (red = 0; red<=Nmax; red++) 
     FgFr[(Nmax-red)*(Nmax+1)+red] += tmp[Nmax*(Nmax+1)+red]*pF[(Nmax+1)*j+Nmax]*a[j];
 }

 /*** SgSr: matrix, SgSr(i,j) = p(Sg = i, Sr = j) ***/

 conv_pF(SgSr, FgFr, Nmax, Bg, Br);

 delete[] FgFr; delete[] tmp;

}
