#include <math.h>
#include "mtrandom.h"
#include <malloc.h>

#define __max(a,b)  (((a) > (b)) ? (a) : (b))
#define __min(a,b)  (((a) < (b)) ? (a) : (b))


extern "C"{

    int calculate1R(double L, double W, double R, int atom_i, double dg,	// linker and grid parameters
                     double* XLocal, double* YLocal, double* ZLocal, 		// atom coordinates
                     double* vdWR, int NAtoms, double vdWRMax,				// v.d.Waals radii
                     double linkersphere, int linknodes, 					// linker routing parameters
                     unsigned char* density)								// returns density array
    {

        // grid
        double x0 = XLocal[atom_i], y0 = YLocal[atom_i], z0 = ZLocal[atom_i];
        int npm = (int)floor(L / dg);
        int ng = 2 * npm + 1, n;
        int ng3 = ng * ng * ng;
        double* xg = new double[ng];
        double* yg = new double[ng];
        double* zg = new double[ng];
        for (int i = -npm; i <= npm; i++)
        {
            n = i + npm;
            xg[n] = i * dg; yg[n] = i * dg; zg[n] = i * dg;
        }

        // select atoms potentially within reach, excluding the attachment point
        double rmaxsq = (L + R + vdWRMax) * (L + R + vdWRMax), rmax, r, rsq, dx, dy, dz;
        int* atomindex = new int[NAtoms];
        int natomsgrid;
        n = 0;
        for (int i = 0; i < NAtoms; i++)
        {
            dx = XLocal[i] - x0; dy = YLocal[i] - y0; dz = ZLocal[i] - z0;
            rsq = dx * dx + dy * dy + dz * dz;
            if ((rsq < rmaxsq) && (i != atom_i)) atomindex[n++] = i;
        }
        natomsgrid = n;
        // local coordinates
        double* xa = new double[natomsgrid];
        double* ya = new double[natomsgrid];
        double* za = new double[natomsgrid];
        double* vdWr = new double[natomsgrid];
        for (int i = 0; i < natomsgrid; i++)
        {
            n = atomindex[i]; vdWr[i] = vdWR[n];
            xa[i] = XLocal[n] - x0; ya[i] = YLocal[n] - y0; za[i] = ZLocal[n] - z0;
        }

        // search for allowed positions
        unsigned char* clash = new unsigned char[ng3];
        for (int i = 0; i < ng3; i++) clash[i] = 0;

        // search for positions causing clashes with atoms
        int ix2, iy2, ir2, rmaxsqint;
        double dx2, dy2, dr2, rmaxsq_dye, rmaxsq_linker;
        int ixmin, ixmax, iymin, iymax, izmin, izmax, offset;
        for (int i = 0; i < natomsgrid; i++)
        {
            rmaxsq_dye = (vdWr[i] + R) * (vdWr[i] + R);
            rmaxsq_linker = (vdWr[i] + 0.5 * W) * (vdWr[i] + 0.5 * W);
            rmax = vdWr[i] + __max(R, 0.5 * W);
            rmaxsq = rmax * rmax;
            ixmin = __max((int)ceil((xa[i] - rmax) / dg), -npm);
            ixmax = __min((int)floor((xa[i] + rmax) / dg), npm);
            iymin = __max((int)ceil((ya[i] - rmax) / dg), -npm);
            iymax = __min((int)floor((ya[i] + rmax) / dg), npm);
            izmin = __max((int)ceil((za[i] - rmax) / dg), -npm);
            izmax = __min((int)floor((za[i] + rmax) / dg), npm);

            for (int ix = ixmin; ix <= ixmax; ix++)
            {
                dx2 = (xg[ix + npm] - xa[i]) * (xg[ix + npm] - xa[i]);
                dy = sqrt(__max(rmaxsq-dx2, 0.));
                iymin = __max((int)ceil((ya[i] - dy) / dg), -npm);
                iymax = __min((int)floor((ya[i] + dy) / dg), npm);
                offset = ng * (ng * (ix + npm) + iymin + npm) + npm;
                for (int iy = iymin; iy <= iymax; iy++)
                {
                    dy2 = (yg[iy + npm] - ya[i]) * (yg[iy + npm] - ya[i]);
                    for (int iz = izmin; iz <= izmax; iz++)
                    {
                        dr2 = dx2 + dy2 + (zg[iz + npm] - za[i]) * (zg[iz + npm] - za[i]);
                        clash[iz + offset] |= (((dr2 <= rmaxsq_dye) << 1) | (dr2 <= rmaxsq_linker));
                    }
                    offset += ng;
                }
            }
        }

        // route linker as a flexible pipe
        double* rlink = new double[ng3];
        double rlink0;
        int ix0, iy0, iz0, linknodes_eff, dlz = 2*linknodes + 1;
        int nnew = 0;
        int* newpos = new int[ng3];

        for (int i = 0; i < ng3; i++) rlink[i] = (clash[i] & 0x01) ? -L : L + L;

        // (1) all positions within linkerinitialsphere*W from the attachment point are allowed
        rmaxsqint = (int)floor(linkersphere * linkersphere * W * W / dg / dg);
        ixmax = __min((int)floor(linkersphere * W / dg), npm);
        n = 0;
        for (int ix = -ixmax; ix <= ixmax; ix++)
        {
            ix2 = ix * ix;
            offset = ng * (ng * (ix + npm) - ixmax + npm) + npm;
            for (int iy = -ixmax; iy <= ixmax; iy++)
            {
                iy2 = iy * iy;
                for (int iz = -ixmax; iz <= ixmax; iz++)
                if (ix2 + iy2 + iz * iz <= rmaxsqint)
                {
                    rlink[iz + offset] = sqrt((double)(ix2 + iy2 + iz * iz)) * dg;
                    newpos[nnew++] = ng * (ng * (npm + ix) + iy + npm) + npm + iz;
                }
                offset += ng;
            }
        }

        // (2) propagate from new positions
        double* sqrts_dg = new double[(2*linknodes*linknodes + 1) * (2*linknodes + 1)];
        for (int ix = 0; ix <= linknodes; ix++)
            for (int iy = 0; iy <= linknodes; iy++)
                for (int iz = -linknodes; iz <= linknodes; iz++)
                    sqrts_dg[(ix*ix + iy*iy)*dlz + iz + linknodes] = sqrt((double)(ix*ix + iy*iy + iz*iz)) * dg;
        while (nnew > 0)
        {
            for (n = 0; n < nnew; n++)
            {
                rlink0 = rlink[newpos[n]];
                linknodes_eff = __min(linknodes, (int)floor((L-rlink0)/dg));
                ix0 = newpos[n]/(ng*ng);
                iy0 = newpos[n]/ng - ix0*ng;
                iz0 = newpos[n] - ix0*ng*ng - iy0*ng;
                ixmin = __max(-linknodes_eff, -ix0);
                ixmax = __min(linknodes_eff, 2 * npm - ix0);
                iymin = __max(-linknodes_eff, -iy0);
                iymax = __min(linknodes_eff, 2 * npm - iy0);
                izmin = __max(-linknodes_eff, -iz0);
                izmax = __min(linknodes_eff, 2 * npm - iz0);

                for (int ix = ixmin; ix <= ixmax; ix++)
                {
                    offset = newpos[n] + ng * (ng * ix + iymin);
                    ix2 = ix * ix;
                    for (int iy = iymin; iy <= iymax; iy++)
                    {
                        ir2 = (ix2 + iy*iy) * dlz + linknodes;
                        for (int iz = izmin; iz <= izmax; iz++)
                        {
                            r = sqrts_dg[ir2 + iz] + rlink0;
                            if ((rlink[iz + offset] > r) && (r < L))
                            {
                                rlink[iz + offset] = r;
                                clash[iz + offset] |= 0x04;
                            }
                        }
                        offset += ng;
                    }
                }
            }

            // update "new" positions
            nnew = 0;
            for (int i = 0; i < ng3; i++)
            {
                if (clash[i] & 0x04) newpos[nnew++] = i;
                clash[i] &= 0x03;
            }
        }

        // search for positions satisfying everything
        n = 0;
        for (int i = 0; i < ng3; i++)
        if (!clash[i] && (rlink[i] <= L))
        {
            density[i] = 1;
            n++;
        }

        delete[] xg; delete[] yg; delete[] zg; delete[] atomindex;
        delete[] xa; delete[] ya; delete[] za; delete[] vdWr;
        delete[] clash; delete[] rlink; delete [] newpos;
        delete[] sqrts_dg;

        return n;
    }

    // AV with three radii
    int calculate3R(double L, double W, double R1, double R2, double R3, int atom_i, double dg,
                     double* XLocal, double* YLocal, double* ZLocal, 		// atom coordinates
                     double* vdWR, int NAtoms, double vdWRMax,				// v.d.Waals radii
                     double linkersphere, int linknodes, 					// linker routing parameters
                     unsigned char* density)								// returns density array
    {

        // grid
        double Rmax = __max(R1, R2); Rmax = __max(Rmax, R3);
        double x0 = XLocal[atom_i], y0 = YLocal[atom_i], z0 = ZLocal[atom_i];
        int npm = (int)floor(L / dg);
        int ng = 2 * npm + 1, n;
        int ng3 = ng * ng * ng;
        double* xg = new double[ng];
        double* yg = new double[ng];
        double* zg = new double[ng];
        for (int i = -npm; i <= npm; i++)
        {
            n = i + npm;
            xg[n] = i * dg; yg[n] = i * dg; zg[n] = i * dg;
        }

        // select atoms potentially within reach, excluding the attachment point
        double rmaxsq = (L + Rmax + vdWRMax) * (L + Rmax + vdWRMax), rmax, r, rsq, dx, dy, dz;
        int* atomindex = new int[NAtoms];
        int natomsgrid;
        n = 0;
        for (int i = 0; i < NAtoms; i++)
        {
            dx = XLocal[i] - x0; dy = YLocal[i] - y0; dz = ZLocal[i] - z0;
            rsq = dx * dx + dy * dy + dz * dz;
            if ((rsq < rmaxsq) && (i != atom_i)) atomindex[n++] = i;
        }
        natomsgrid = n;
        // local coordinates
        double* xa = new double[natomsgrid];
        double* ya = new double[natomsgrid];
        double* za = new double[natomsgrid];
        double* vdWr = new double[natomsgrid];
        for (int i = 0; i < natomsgrid; i++)
        {
            n = atomindex[i]; vdWr[i] = vdWR[n];
            xa[i] = XLocal[n] - x0; ya[i] = YLocal[n] - y0; za[i] = ZLocal[n] - z0;
        }

        // search for allowed positions
        unsigned char* clash = new unsigned char[ng3];
        for (int i = 0; i < ng3; i++) clash[i] = 0;

        // search for positions causing clashes with atoms
        int ix2, iy2, ir2, rmaxsqint;
        double dx2, dy2, dr2, rmaxsq_dye1, rmaxsq_dye2, rmaxsq_dye3, rmaxsq_linker;
        int ixmin, ixmax, iymin, iymax, izmin, izmax, offset;
        for (int i = 0; i < natomsgrid; i++)
        {
            rmaxsq_dye1 = (vdWr[i] + R1) * (vdWr[i] + R1);
            rmaxsq_dye2 = (vdWr[i] + R2) * (vdWr[i] + R2);
            rmaxsq_dye3 = (vdWr[i] + R3) * (vdWr[i] + R3);
            rmaxsq_linker = (vdWr[i] + 0.5 * W) * (vdWr[i] + 0.5 * W);
            rmax = vdWr[i] + __max(Rmax, 0.5 * W);
            rmaxsq = rmax * rmax;
            ixmin = __max((int)ceil((xa[i] - rmax) / dg), -npm);
            ixmax = __min((int)floor((xa[i] + rmax) / dg), npm);
            iymin = __max((int)ceil((ya[i] - rmax) / dg), -npm);
            iymax = __min((int)floor((ya[i] + rmax) / dg), npm);
            izmin = __max((int)ceil((za[i] - rmax) / dg), -npm);
            izmax = __min((int)floor((za[i] + rmax) / dg), npm);

            for (int ix = ixmin; ix <= ixmax; ix++)
            {
                dx2 = (xg[ix + npm] - xa[i]) * (xg[ix + npm] - xa[i]);
                dy = sqrt(__max(rmaxsq-dx2, 0.));
                iymin = __max((int)ceil((ya[i] - dy) / dg), -npm);
                iymax = __min((int)floor((ya[i] + dy) / dg), npm);
                offset = ng * (ng * (ix + npm) + iymin + npm) + npm;
                for (int iy = iymin; iy <= iymax; iy++)
                {
                    dy2 = (yg[iy + npm] - ya[i]) * (yg[iy + npm] - ya[i]);
                    for (int iz = izmin; iz <= izmax; iz++)
                    {
                        dr2 = dx2 + dy2 + (zg[iz + npm] - za[i]) * (zg[iz + npm] - za[i]);
                        clash[iz + offset] |= (((dr2 <= rmaxsq_dye3) << 3) | ((dr2 <= rmaxsq_dye2) << 2) |
                            ((dr2 <= rmaxsq_dye1) << 1) | (dr2 <= rmaxsq_linker));
                    }
                    offset += ng;
                }
            }
        }

        // route linker as a flexible pipe
        double* rlink = new double[ng3];
        double rlink0;
        int ix0, iy0, iz0, linknodes_eff, dlz = 2*linknodes + 1;
        int nnew = 0;
        int* newpos = new int[ng3];

        for (int i = 0; i < ng3; i++) rlink[i] = (clash[i] & 0x01) ? -L : L + L;

        // (1) all positions within linkerinitialsphere*W from the attachment point are allowed
        rmaxsqint = (int)floor(linkersphere * linkersphere * W * W / dg / dg);
        ixmax = __min((int)floor(linkersphere * W / dg), npm);
        n = 0;
        for (int ix = -ixmax; ix <= ixmax; ix++)
        {
            ix2 = ix * ix;
            offset = ng * (ng * (ix + npm) - ixmax + npm) + npm;
            for (int iy = -ixmax; iy <= ixmax; iy++)
            {
                iy2 = iy * iy;
                for (int iz = -ixmax; iz <= ixmax; iz++)
                if (ix2 + iy2 + iz * iz <= rmaxsqint)
                {
                    rlink[iz + offset] = sqrt((double)(ix2 + iy2 + iz * iz)) * dg;
                    newpos[nnew++] = ng * (ng * (npm + ix) + iy + npm) + npm + iz;
                }
                offset += ng;
            }
        }

        // (2) propagate from new positions
        double* sqrts_dg = new double[(2*linknodes*linknodes + 1) * (2*linknodes + 1)];
        for (int ix = 0; ix <= linknodes; ix++)
            for (int iy = 0; iy <= linknodes; iy++)
                for (int iz = -linknodes; iz <= linknodes; iz++)
                    sqrts_dg[(ix*ix + iy*iy)*dlz + iz + linknodes] = sqrt((double)(ix*ix + iy*iy + iz*iz)) * dg;
        while (nnew > 0)
        {
            for (n = 0; n < nnew; n++)
            {
                rlink0 = rlink[newpos[n]];
                linknodes_eff = __min(linknodes, (int)floor((L-rlink0)/dg));
                ix0 = newpos[n]/(ng*ng);
                iy0 = newpos[n]/ng - ix0*ng;
                iz0 = newpos[n] - ix0*ng*ng - iy0*ng;
                ixmin = __max(-linknodes_eff, -ix0);
                ixmax = __min(linknodes_eff, 2 * npm - ix0);
                iymin = __max(-linknodes_eff, -iy0);
                iymax = __min(linknodes_eff, 2 * npm - iy0);
                izmin = __max(-linknodes_eff, -iz0);
                izmax = __min(linknodes_eff, 2 * npm - iz0);

                for (int ix = ixmin; ix <= ixmax; ix++)
                {
                    offset = newpos[n] + ng * (ng * ix + iymin);
                    ix2 = ix * ix;
                    for (int iy = iymin; iy <= iymax; iy++)
                    {
                        ir2 = (ix2 + iy*iy) * dlz + linknodes;
                        for (int iz = izmin; iz <= izmax; iz++)
                        {
                            r = sqrts_dg[ir2 + iz] + rlink0;
                            if ((rlink[iz + offset] > r) && (r < L))
                            {
                                rlink[iz + offset] = r;
                                clash[iz + offset] |= 0x10;
                            }
                        }
                        offset += ng;
                    }
                }
            }

            // update "new" positions
            nnew = 0;
            for (int i = 0; i < ng3; i++)
            {
                if (clash[i] & 0x10) newpos[nnew++] = i;
                clash[i] &= 0x0F;
            }
        }

        // search for positions satisfying everything
        n = 0; int dn = 0;
        for (int i = 0; i < ng3; i++)
        {
            if ((clash[i] & 0x01) || rlink[i] > L) continue;
            dn = ((~clash[i] & 0x08) >> 3) + ((~clash[i] & 0x04) >> 2) + ((~clash[i] & 0x02) >> 1);
            density[i] = dn;
            n += dn;
        }

        delete[] xg; delete[] yg; delete[] zg; delete[] atomindex;
        delete[] xa; delete[] ya; delete[] za; delete[] vdWr;
        delete[] clash; delete[] rlink; delete [] newpos;
        delete[] sqrts_dg;

        return n;
    }

    // average distances
    const int RMEAN_MINSAMPLE = 1;
    const int RMEAN_MAXSAMPLE = 64;

    struct vector3
    {
       double x;
       double y;
       double z;
    };

    // <RDA>
    double rdamean(vector3* av1, int av1length, vector3* av2, int av2length, int nsamples, int rndseed)
    {
        MTrandoms rmt;
        rmt.seedMT(rndseed);
        double dx, dy, dz, r = 0.;
        int i1, i2;
        int jmax = __min(RMEAN_MAXSAMPLE, RMEAN_MINSAMPLE+__min(av1length, av2length)/100), imax = nsamples/jmax;
        for (int i=0; i<imax; i++) {
            i1 = (int)(rmt.random0i1e() * (av1length-jmax+1));
            i2 = (int)(rmt.random0i1e() * (av2length-jmax+1));
            for (int j=0; j<jmax; j++) {
                dx = av1[i1+j].x - av2[i2+j].x;
                dy = av1[i1+j].y - av2[i2+j].y;
                dz = av1[i1+j].z - av2[i2+j].z;
                r += sqrt(dx*dx + dy*dy + dz*dz);
            }
        }
        return r/((double)(imax*jmax));
    }

    // <RDA>E
    double rdameanE(vector3* av1, int av1length, vector3* av2, int av2length, int nsamples, int rndseed, double R0)
    {
        MTrandoms rmt;
        rmt.seedMT(rndseed);
        double dx, dy, dz, r2, e = 0., R0r6 = 1./(R0*R0*R0*R0*R0*R0);
        int i1, i2;
        int jmax = __min(RMEAN_MAXSAMPLE, RMEAN_MINSAMPLE+__min(av1length, av2length)/100), imax = nsamples/jmax;
        for (int i=0; i<imax; i++) {
            i1 = (int)(rmt.random0i1e() * (av1length-jmax+1));
            i2 = (int)(rmt.random0i1e() * (av2length-jmax+1));
            for (int j=0; j<jmax; j++) {
                dx = av1[i1+j].x - av2[i2+j].x;
                dy = av1[i1+j].y - av2[i2+j].y;
                dz = av1[i1+j].z - av2[i2+j].z;
                r2 = dx*dx + dy*dy + dz*dz;
                e += 1. / (1. + r2 * r2 * r2 * R0r6);
            }
        }
        e /= (double)(imax*jmax);
        return R0 * pow((1./e - 1.), 1./ 6.);
    }}
