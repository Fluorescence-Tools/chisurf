// Constants as IDG2 - the inverse squared grid distance
// DT the time step
// NG, NG2 the number of grid points and grid points squared
// have to be defined beforehand

__kernel void copy3d(__global float *n,           // New iteration
                     __global const float *p,      // Previous iteration
                     __global const uchar *b      // Previous iteration
                    )
    {
    int i = get_global_id(0);
    if (b[i]>0){
        n[i] = p[i];
    }
}


__kernel void iterate(__global float *n,            // New iteration
                      __global const float *p,      // Previous iteration
                      __global const uchar *b,      // bounds
                      __global const float *d,      // diffusion coefficients
                      __global const float *k       // rate constants of (de)excitation
                      )
    {
    const uint  ix = get_global_id(0) + 1;
    const uint  iy = get_global_id(1) + 1;
    const uint  iz = get_global_id(2) + 1;

    const uint i   = (ix + 0) + NG * (iy + 0) + NG2 * (iz + 0);// the current index

    const uint ixl = (ix - 1) + NG * (iy + 0) + NG2 * (iz + 0);// x left
    const uint ixr = (ix + 1) + NG * (iy + 0) + NG2 * (iz + 0);// x right

    const uint iyl = (ix + 0) + NG * (iy - 1) + NG2 * (iz + 0);
    const uint iyr = (ix + 0) + NG * (iy + 1) + NG2 * (iz + 0);

    const uint izl = (ix + 0) + NG * (iy + 0) + NG2 * (iz - 1);
    const uint izr = (ix + 0) + NG * (iy + 0) + NG2 * (iz + 1);

    if (b[i]>0){
        //Definition of the neighboring cells
        float dp = d[i] * p[i];
        float xl = (dp - d[ixl] * p[ixl]) * b[ixl];// * IDG2 * DT;
        float xr = (dp - d[ixr] * p[ixr]) * b[ixr];// * IDG2 * DT;

        float yl = (dp - d[iyl] * p[iyl]) * b[iyl];// * IDG2 * DT;
        float yr = (dp - d[iyr] * p[iyr]) * b[iyr];// * IDG2 * DT;

        float zl = (dp - d[izl] * p[izl]) * b[izl];// * IDG2 * DT;
        float zr = (dp - d[izr] * p[izr]) * b[izr];// * IDG2 * DT;

        float ts = k[i] * p[i];// * DT;

        n[i] -= (xl + xr + yl + yr + zl + zr + ts);//(tr + tb + ta + ts);
        //n[i] -= (tr + tb + ta + ts);
        n[ixr] += xr; n[ixl] += xl;
        n[iyr] += yr; n[iyl] += yl;
        n[izr] += zr; n[izl] += zl;
    }
}


__kernel
void sumGPU(
    __global const float *input,
    __global float *partialSums,
    uint it,
    uint local_size,
    __local float *localSums)
 {
    // example from https://dournac.org/info/gpu_sum_reduction
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint global_id = get_global_id(0);
    // Copy from global memory to local memory
    localSums[local_id] = (local_id < global_id) ? input[global_id] : 0;
    // Loop for computing localSums
    for (uint stride = group_size/2; stride>0; stride/=2){
        // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);
        // Divide WorkGroup into 2 parts and add elements 2 by 2
        // between local_id and local_id + stride
        if (local_id < stride)
            localSums[local_id] += localSums[local_id + stride];
    }
    // Write result into partialSums[nWorkGroups]
    if (local_id == 0)
        partialSums[local_size * it + get_group_id(0)] = localSums[0];
 }

/* slow (direct) convolution */
__kernel
void sconv(__global float *fit, __global float *decay, __global float *lamp, float dt)
{
    int i = get_global_id(0);
    int j = 1;
    /* convolution */
    fit[i] = 0.5 * lamp[0] * decay[i];
    for (int j=1; j<i; j++)
        fit[i] += lamp[j] * decay[i-j];
    fit[i] += 0.5 * lamp[i] * decay[0];
    fit[i] = fit[i] * dt;
}