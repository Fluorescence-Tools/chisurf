/* extern "C" wrappers around the burbulator C++ functions.
   Prevents C++ name-mangling so they can be loaded by name via
   ctypes on all platforms, not just Windows (where the .def file
   already handles this). */

#include "smdif.h"

extern "C" {

unsigned long smdif_ov3_c(
    int N_species,
    double* M, double* D,
    int N_channels,
    double* q, double* q_bg,
    double* k_rad, double* k_nrad,
    double box_xy, double box_z,
    int focus_type, double* focus_param,
    double dt,
    unsigned long N_ph_max,
    unsigned long* data_T, double* data_t,
    short* data_N, short* data_species,
    int* data_molecule,
    unsigned long* T0, int* N_molecules,
    double* x, double* y, double* z,
    short int* species,
    int rmt1seed,
    unsigned long* rmt1state, int* rmt1left,
    int rmt2seed,
    unsigned long* rmt2state, int* rmt2left)
{
    return smdif_ov3(
        N_species, M, D, N_channels,
        q, q_bg, k_rad, k_nrad,
        box_xy, box_z,
        focus_type, focus_param,
        dt, N_ph_max,
        data_T, data_t, data_N, data_species, data_molecule,
        *T0, *N_molecules,
        x, y, z, species,
        rmt1seed, rmt1state, *rmt1left,
        rmt2seed, rmt2state, *rmt2left
    );
}

int data2spc132_tac_c(
    int pulsed_exc,
    int N_channels,
    unsigned long* data_T, double* data_t,
    short* data_N, short* data_species,
    int* data_molecule,
    unsigned long N_photons,
    double tw,
    unsigned short* ch_conversion,
    int N_tac_channels,
    double tac_dt, double laser_period,
    double* F, int* lookup,
    char* spc_data,
    unsigned long* MT_ov, unsigned long* i,
    unsigned long* rmt2state, int* rmt2left)
{
    return data2spc132_tac(
        pulsed_exc, N_channels,
        data_T, data_t, data_N, data_species, data_molecule,
        N_photons, tw,
        ch_conversion,
        N_tac_channels, tac_dt, laser_period,
        F, lookup,
        spc_data,
        *MT_ov, *i,
        rmt2state, *rmt2left
    );
}

unsigned long rotdiff_c(
    double D, double tau, double r0,
    double l1, double l2,
    double q,
    double dt,
    unsigned long N_ph_max,
    unsigned long* data_T, double* data_t,
    short* data_N, short* data_species,
    int* data_molecule,
    unsigned long* T0,
    double* x, double* y, double* z,
    int rmt1seed,
    unsigned long* rmt1state, int* rmt1left,
    int rmt2seed,
    unsigned long* rmt2state, int* rmt2left)
{
    return rotdiff(
        D, tau, r0, l1, l2, q,
        dt, N_ph_max,
        data_T, data_t, data_N, data_species, data_molecule,
        *T0,
        *x, *y, *z,
        rmt1seed, rmt1state, *rmt1left,
        rmt2seed, rmt2state, *rmt2left
    );
}

} /* extern "C" */
