#include <math.h>
#include "mtrandom.h"


// sorting from numerical recepies, modified for 2 arrays

void shell2(unsigned long n, double* a, int* b)
/*Sorts an array a[] into ascending numerical order by Shell’s method (diminishing increment
sort). a is replaced on output by its sorted rearrangement. Normally, the argument n should
be set to the size of array a, but if n is smaller than this, then only the first n elements of a
are sorted. This feature is used in selip.*/

{
unsigned long i,j,inc;
double v;
int w;
inc=1; 				// Determine the starting increment.

do {
  inc *= 3;
  inc++;
  } while (inc <= n);

do { 				//Loop over the partial sorts.
  inc /= 3;
  for (i=inc;i<n;i++) { 	//Outer loop of straight insertion.
    v=a[i];
    w=b[i];
    j=i;
    while (a[j-inc] > v) { 	//Inner loop of straight insertion.
      a[j]=a[j-inc];
      b[j]=b[j-inc];
      j -= inc;
      if (j < inc) break;
    }
    a[j]=v;
    b[j]=w;
  }
} while (inc > 1);

}


// sorting from numerical recepies, modified for 4 arrays

void shell4(unsigned long n, double* a, int* b, int* c, int* d)

{
unsigned long i,j,inc;
double v;
int w1,w2,w3;
inc=1; 				// Determine the starting increment.

do {
  inc *= 3;
  inc++;
  } while (inc <= n);

do { 				//Loop over the partial sorts.
  inc /= 3;
  for (i=inc;i<n;i++) { 	//Outer loop of straight insertion.
    v=a[i];
    w1=b[i];
    w2=c[i];
    w3=d[i];
    j=i;
    while (a[j-inc] > v) { 	//Inner loop of straight insertion.
      a[j]=a[j-inc];
      b[j]=b[j-inc];
      c[j]=c[j-inc];
      d[j]=d[j-inc];
      j -= inc;
      if (j < inc) break;
    }
    a[j]=v;
    b[j]=w1;
    c[j]=w2;
    d[j]=w3;
  }
} while (inc > 1);

}

// sorting in reversed order

void shell_r(unsigned long n, int* a)

{
unsigned long i,j,inc;
int v;
inc=1; 				// Determine the starting increment.

do {
  inc *= 3;
  inc++;
  } while (inc <= n);

do { 				//Loop over the partial sorts.
  inc /= 3;
  for (i=inc;i<n;i++) { 	//Outer loop of straight insertion.
    v=a[i];
    j=i;
    while (a[j-inc] < v) { 	//Inner loop of straight insertion.
      a[j]=a[j-inc];
      j -= inc;
      if (j < inc) break;
    }
    a[j]=v;
  }
} while (inc > 1);

}
/*  qnorm.c    CCMATH mathematics library source code.
 *
 *  Copyright (C)  2000   Daniel A. Atkinson    All rights reserved.
 *  This code may be redistributed under the terms of the GNU library
 *  public license (LGPL). ( See the lgpl.license file for details.)
 * ------------------------------------------------------------------------
 */
/*  qnorm

     Integral from x to infinity of the standard normal distribution.

     double qnorm(double x)
       x = value of argument
      return value: Qn(x) = integral of normal density from x to infinity

*/
/* !!! standard normal distribution = 1/sqrt(2pi)*exp(-x^2/2) !!! => 
       qnorm = 1/2*erfc(x/sqrt(2)) */

double qnorm(double x)
{ double y,ro,f,t; int k,nf;
  if(x<0.){ x= -x; nf=0;} else nf=1;
  y=x*x; ro=exp(-y/2.)/2.506628274631;
  if(x<3.){ f=t=1.;
    for(k=1; t>1.e-14 ;){ t*=y/(k+=2); f+=t;}
    f=.5-x*ro*f; }
  else{ f=x; k=ceil(250./y); if(k<3) k=3;
    for(; k>0 ;) f=x+(k--)/f;
    f=ro/f; }
  if(nf) return f; else return 1.-f;
}

/* random number, p(x) ~ erf(x/sqrt(2)) */
/* (maybe rather unefficient) */

double random_erfc(MTrandoms& rmt)
{
  const double sqrt_pi_half = 1.2533141373155;
  double v,x,y;

  do {
    /* exp random number */
    v = rmt.random0e1e();
    x = -log(v)*sqrt_pi_half;

    /* uniform random number 0..x */
    y = v * rmt.random0i1e();
  }
  while (y>2.*qnorm(x));

  return x;
}

