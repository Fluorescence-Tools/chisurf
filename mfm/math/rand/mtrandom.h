// Random number generators header.
// see mtrandom.cpp.
// Vesion 2010-07-30

#include <math.h>
#define N 624

const double f1 = 2.3283064365387E-010;
const double f2 = 2.3283064370808E-010;
const double sqrt2en = 3.994274348768903E-010;

class MTrandoms
{

public:

 MTrandoms ();
 MTrandoms& operator = (const MTrandoms&);

 void seedMT(unsigned int);
 void seedMT(void);
 void init_by_array(unsigned int*, int);

 void GetState(unsigned int*, int&, unsigned int*);
 void SetState(unsigned int*, int, unsigned int*);

 double random_res53();

/* generates a random number on [0,0xffffffff]-interval */
unsigned int inline randomUInt(void)
{
    unsigned int y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    return (y ^(y >> 18));
}

/* generates a random number on [-...,0x7fffffff]-interval */
int inline randomInt(void)
{
    unsigned int y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    return (y ^(y >> 18));
}

/* generates a random number on [0,0x7fffffff]-interval */
int inline randomIntPlus(void)
{
    unsigned int y;

    if (--left == 0) next_state();
    y = *next++;

    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);
    return (y >> 1);
}
/* generates a random number on [0,1]-real-interval */
double inline random0i1i(void)
{
    unsigned int y;

    if (--left == 0) next_state();
    y = *next++;

    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;

    return (y ^(y >> 18)) * f2; 
}


/* generates a random number on (0,1)-real-interval */
double inline random0e1e(void)
{
    unsigned int y;

    if (--left == 0) next_state();
    y = *next++;

    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;

    return ((y ^ (y >> 18)) + 0.5) * f1; 
}

/* generates a random number on [0,1)-real-interval */
double inline random0i1e(void)
{
    unsigned int y;

    if (--left == 0) next_state();
    y = *next++;

    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;

    return ((y ^(y >> 18)) * f1); 
}

/* generates a random number on - sqrt(2/e) < v < sqrt(2/e) */
double inline random4nrm(void)
{
    unsigned int y;

    if (--left == 0) next_state();
    y = *next++;
    
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

    return ((long)y * sqrt2en); 
}

double randomNorm();

private:

 unsigned int state[N]; /* the array for the state vector  */
 int left;
 unsigned int *next;

 void next_state(void);

};
