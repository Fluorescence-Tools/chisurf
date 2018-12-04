/* sgsr.cpp */
void sgsr_pN(double*, double*, unsigned int, double, double, double);
void sgsr_pF(double*, double*, unsigned int, double, double, double);
void sgsr_pN_manypg (double*, double*, unsigned int, double, double, unsigned int, double*, double*);
void sgsr_pF_manypg (double*, double*, unsigned int, double, double, unsigned int, double*, double*);
void sgsr_manypF (double*, double*, unsigned int, double, double, unsigned int, double*, double*);
void conv_pF (double*, double*, unsigned int, double, double);


/* sgsr_misc.cpp */
void polynom2_conv (double*, double*, unsigned int, double);
void poisson_0toN (double*, double, unsigned int);
void poisson_0toN_multi (double*, double*, unsigned int, unsigned int);
