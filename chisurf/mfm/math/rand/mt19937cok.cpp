/* 
   A C-program for MT19937, with initialization improved 2002/2/10.
   Coded by Takuji Nishimura and Makoto Matsumoto.
   This is a faster version by taking Shawn Cokus's optimization,
   Matthe Bellew's simplification, Isaku Wada's real version.

   Before using, initialize the state by using init_genrand(seed) 
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

/* Period parameters */  
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

#include "mtrandom.h"
#include <time.h>

MTrandoms::MTrandoms ()
 {
  left = 1;
  seedMT((unsigned long)(time(0)));
 }

MTrandoms& MTrandoms::operator = (const MTrandoms& b)
 {
  if (this==&b) return *this;
  for (int j=0; j<N; j++) state[j] = b.state[j];	
  next = state + (b.next - b.state);		
  left = b.left;	
  return *this;		
 }

void MTrandoms::GetState(unsigned int* lstate, int& lleft, unsigned int* lnext)
{
  for(int j = 0; j < N; j++) lstate[j] = state[j];
  lleft = left;
  lnext = next;
}

void MTrandoms::SetState(unsigned int* lstate, int lleft, unsigned int* lnext)
{
  for(int j = 0; j < N; j++) state[j] = lstate[j];
  left = lleft;
  next = lnext;
}

void MTrandoms::seedMT ()
 {
    seedMT((unsigned int)(time(0)));
 }

/* initializes state[N] with a seed */
void MTrandoms::seedMT(unsigned int s)
{
    int j;
    state[0]= s & 0xffffffffUL;
    for (j=1; j<N; j++) {
        state[j] = (1812433253UL * (state[j-1] ^ (state[j-1] >> 30)) + j); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array state[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        state[j] &= 0xffffffffUL;  /* for >32 bit machines */
    }
    left = 1;
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void MTrandoms::init_by_array(unsigned int init_key[], int key_length)
{
    int i, j, k;
    MTrandoms::seedMT(19650218UL);
    i=1; j=0;
    k = (N>key_length ? N : key_length);
    for (; k; k--) {
        state[i] = (MTrandoms::state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; /* non linear */
        state[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=N) { state[0] = state[N-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=N-1; k; k--) {
        state[i] = (state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        state[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=N) { state[0] = state[N-1]; i=1; }
    }

    state[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
    left = 1;
}

void MTrandoms::next_state(void)
{
    unsigned int *p=state;
    int j;

    left = N;
    next = state;
    
    for (j=N-M+1; --j; p++) 
        *p = p[M] ^ TWIST(p[0], p[1]);

    for (j=M; --j; p++) 
        *p = p[M-N] ^ TWIST(p[0], p[1]);

    *p = p[M-N] ^ TWIST(p[0], state[0]);
}


/* generates a random number on [0,1) with 53-bit resolution*/
double MTrandoms::random_res53(void) 
{ 
    unsigned int a=MTrandoms::randomUInt()>>5, b=MTrandoms::randomUInt()>>6; 
    return(a*67108864.0+b)*(1.0/9007199254740992.0); 
} 
/* These real versions are due to Isaku Wada, 2002/01/09 added */


double MTrandoms::randomNorm()
{
     double u, v, q, x1, x2;
	
     const double ei = 0.27597, eo = 0.27846;
     const double sqrt2e = 1.71552776992141, a = 0.449871, b = 0.386595;

     for(;;)
	{
          // Generate P = (u,v) uniform in rectangle enclosing
          // acceptance region:
          //   0 < u < 1
          // - sqrt(2/e) < v < sqrt(2/e)
          // The constant below is 2*sqrt(2/e).

          u = random0i1e();
          //v = sqrt2e * (random0i1e() - 0.5);
          v = random4nrm();
		  
          // Evaluate the quadratic form
	      x1 = u - a;
  	      x2 = fabs(v) + b;
 	      q = x1*x1 + (0.19600*x2 - 0.25472*x1)*x2;

          // Accept P if inside inner ellipse
          if (q < ei) break;
	   
          // Reject P if outside outer ellipse
          if (q > eo) continue;
	 
          // Between ellipses: perform exact test
          if (v*v <= -4.0 * log(u)*u*u) break;
        } 
	
     return v/u;
}
