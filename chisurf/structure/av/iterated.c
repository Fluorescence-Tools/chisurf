// Constants as IDG2 - the inverse squared grid distance
// NG, NG2 the number of grid points and grid points squared
// have to be defined beforehand

__kernel void copy3d(__global float *target,
                     __global const float *source,
                     __global const uchar *mask
                    )
    {
    int i = get_global_id(0);
    if (mask[i]>0){
        target[i] = source[i];
    }
}

__kernel void multiply(__global const float *a_g,
                       __global const float *b_g,
                       __global float *res_g,
                       __global const uchar *mask
                       )
{
  int gid = get_global_id(0);
  if (mask[gid]>0){
      res_g[gid] = a_g[gid] * b_g[gid];
  }
}


__kernel void sum(__global const float *a_g,
                  __global const float *b_g,
                  __global float *res_g,
                  __global const uchar *mask
                 )
{
  int gid = get_global_id(0);
  if (mask[gid]>0){
      res_g[gid] = a_g[gid] + b_g[gid];
  }
}



__kernel void iterate(__global float *n,            // New iteration
                      __global const float *p,      // Previous iteration
                      __global const float *d,      // diffusion coefficients
                      __global const float *k,       // rate constants of (de)excitation
                      __global const uchar *b      // bounds
                      )
    {
    const uint ix = get_global_id(0) + 1;
    const uint iy = get_global_id(1) + 1;
    const uint iz = get_global_id(2) + 1;
    const uint i   = (ix + 0) + NG * (iy + 0) + NG2 * (iz + 0);// the current index

    if (b[i]>0){
        const uint ixl = (ix - 1) + NG * (iy + 0) + NG2 * (iz + 0);// x left
        const uint ixr = (ix + 1) + NG * (iy + 0) + NG2 * (iz + 0);// x right
        const uint iyl = (ix + 0) + NG * (iy - 1) + NG2 * (iz + 0);
        const uint iyr = (ix + 0) + NG * (iy + 1) + NG2 * (iz + 0);
        const uint izl = (ix + 0) + NG * (iy + 0) + NG2 * (iz - 1);
        const uint izr = (ix + 0) + NG * (iy + 0) + NG2 * (iz + 1);

        const float dp = d[i] * p[i];
        const float xl = (dp - d[ixl] * p[ixl]) * b[ixl];
        const float xr = (dp - d[ixr] * p[ixr]) * b[ixr];
        const float yl = (dp - d[iyl] * p[iyl]) * b[iyl];
        const float yr = (dp - d[iyr] * p[iyr]) * b[iyr];
        const float zl = (dp - d[izl] * p[izl]) * b[izl];
        const float zr = (dp - d[izr] * p[izr]) * b[izr];

        const float ts = k[i] * p[i];

        n[i] = p[i] - (xl + xr + yl + yr + zl + zr + ts);

    }
}


__kernel
void reduce_decay(
            __global float* p,
            __global float* k,
            __local float* scratch_d,
            __local float* scratch_p,
            __const int length,
            __const int n_local,
            __const int i_out,
            __const float t,
            __global float* result_d,
            __global float* result_p) {

  int global_index = get_global_id(0);
  float accumulator_d = 0;
  float accumulator_p = 0;
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    accumulator_d += p[global_index] * exp(-t * k[global_index]);
    accumulator_p += p[global_index];
    global_index += get_global_size(0);
  }

  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratch_d[local_index] = accumulator_d;
  scratch_p[local_index] = accumulator_p;

  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
    if (local_index < offset) {
      scratch_d[local_index] += scratch_d[local_index + offset];
      scratch_p[local_index] += scratch_p[local_index + offset];
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  }
  if (local_index == 0) {
    result_d[i_out * n_local + get_group_id(0)] = scratch_d[0];
    result_p[i_out * n_local + get_group_id(0)] = scratch_p[0];
  }
}