// misc. functions for SgSr program

#include <math.h>

// convolves vectors p and [p2 1-p2]
void polynom2_conv (double* return_p, double* p, unsigned int n, double p2)
{
  unsigned int i;
  return_p[0] = p[0]*p2;
  for (i = 1; i<n; i++) return_p[i] = p[i-1]*(1.-p2) + p[i]*p2;
  return_p[n] = p[n-1]*(1.-p2);
}

// generates Poisson distribution witn average= lambda, for 0..N
void poisson_0toN (double* return_p, double lambda, unsigned int N)
{
  unsigned int i;
  return_p[0] = exp(-lambda);
  for (i = 1; i<=N; i++) return_p[i] = return_p[i-1]*lambda/(double)i;
}

// generates Poisson distribution for a set of lambdas
void poisson_0toN_multi (double* return_p, double* lambda, unsigned int M, unsigned int N)
{
  unsigned int i, j, ishift;
  for (j = 0; j<M; j++) {
    ishift = (N+1)*j;
    return_p[ishift] = exp(-lambda[j]);
    for (i = 1; i<=N; i++) return_p[ishift+i] = return_p[ishift+i-1]*lambda[j]/(double)i;
  }
}
