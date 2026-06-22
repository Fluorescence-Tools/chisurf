// data_T, data_t, data_N --> spc132 format, with TAC
// version compatible with C# BinaryWriter

#include <math.h>
#include "mtrandom.h"

int data2spc132_tac(

 int pulsed_exc,		// pulsed excitation on?
 int N_channels,		// number of detection channels
 unsigned long* data_T,		// number of time window
 double* data_t,		// times of photon arrival
 short* data_N,			// number of channel for each photon
 short* data_species,		// emitting species #
 int* data_molecule,		// emitting molecule #

 unsigned long N_photons,	// number of photons to be converted
 double tw,			// original time window
 unsigned short* ch_conversion,	// i-th channel -> ch_conversion[i]

 int N_tac_channels,		// number of TAC channels
 double tac_dt,			// time per channel
 double laser_period,		// laser period
 double* F,			// integrated p(t)
 int* lookup,			// lookup tables

 char* spc_data,		// output
 unsigned long& MT_ov,		// total number of MT overflows
 unsigned long& i,		// first record in spc_data

 unsigned long* rmt2state,	// emission generator
 int& rmt2left

)

{

 const double SYNC_DT = laser_period*1.e-6;
 const int MT_MAX_N = 0xfff;
 const int BYTEMASK = 0xff;
 const double MT_MAX_T = (double)(MT_MAX_N+1)*SYNC_DT;
 const double TAC_CALIB = tac_dt*1.e-6;

 /* local */
 unsigned long n = 0;

 int MT,			// macro time
 tac,				// micro time
 N_spc,				// SPC132 channel number
 MT_ov_last;			// since last photon

 unsigned long i_shift,		// index shift for F and lookup
 data_T_prev = data_T[0];	// previous data_T

 double t, t_offset = 0.,	// time, offset due to data_T overflows
 r;				// a random number

 MTrandoms rmt2;		// random number generator
 rmt2.SetState(rmt2state, rmt2left);

 while (t_offset + tw*(double)data_T[0] + data_t[0] < ((double)(MT_ov)-1.)*MT_MAX_T)
   t_offset += tw*4294967296L;

 do {

   if (data_T[n] < data_T_prev) t_offset += tw*4294967296L; // overflow
   t = t_offset + tw*(double)data_T[n] + data_t[n];
   MT = ceil((t-(double)MT_ov*MT_MAX_T)/SYNC_DT);

   if (MT>MT_MAX_N) {

     /* invalid photon */
     MT_ov_last = MT/(MT_MAX_N+1);

     spc_data[i++] = MT_ov_last & BYTEMASK;
     spc_data[i++] = (MT_ov_last >> 8) & BYTEMASK;
     spc_data[i++] = (MT_ov_last >> 16) & BYTEMASK;
     spc_data[i++] = ((MT_ov_last >> 24) & 0x0f) + 0xc0;

     MT_ov += MT_ov_last;
     MT -= MT_ov_last*(MT_MAX_N+1);
   }

   /* valid photon */
   if (pulsed_exc) {
     /** TAC **/
     i_shift = (data_species[n]*N_channels + data_N[n])*N_tac_channels;
     r = rmt2.random0i1e();
     tac = lookup[i_shift + int(floor(r*N_tac_channels))]; 
     while (F[i_shift + tac] < r) tac++;
     /* invert */
     tac = N_tac_channels - tac - 1;
   }
   else /* cw */ 
     //tac = floor(((double)MT_ov*MT_MAX_T + (double)MT*SYNC_DT - t)/TAC_CALIB); // sometimes causes tac=-1
     tac = floor((SYNC_DT-fmod(t,SYNC_DT))/TAC_CALIB);

   // convert channel numbers
   N_spc = ch_conversion[data_N[n]];

   spc_data[i++] = MT & BYTEMASK;
   spc_data[i++] = (N_spc << 4) + (MT >> 8);
   spc_data[i++] = tac & BYTEMASK;
   spc_data[i++] = (tac >> 8);

   data_T_prev = data_T[n];
   n++;
 }
 while (n<N_photons);
 
 rmt2.GetState(rmt2state, rmt2left);

return 1;
}

// cl /TP /O2 /EHsc /MD /Fedata2spc_tac.dll data2spc_tac.cpp mt19937cok.cpp /link /dll /export:data2spc132_tac
// VC8: mt /manifest data2spc_tac.dll.manifest /outputresource:"data2spc_tac.dll;#2"