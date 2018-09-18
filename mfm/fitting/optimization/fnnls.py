#!/usr/bin/env python

#
# ported fnnls.m to fnnls.py
#
# gjok - 20050816
#
# TODO: DOESNT WORK!!!!!

import numpy
import numpy.linalg.linalg as la


def any(X)     : return len(numpy.nonzero(X)) != 0
def find(X)    : return numpy.nonzero(X)

def norm(X, d) :
    return max(numpy.sum(abs(X), axis=1))

#
# x, w = fnnls(XtX, Xty, tol)
#
eps=1e-10

def fnnls(XtX, Xty, tol=0):
    '''
    #FNNLS	Non-negative least-squares.
    #
    # 	Adapted from NNLS of Mathworks, Inc.
    #          [x,w] = nnls(X, y)
    #
    #	x, w = fnnls(XtX,Xty) returns the vector X that solves x = pinv(XtX)
        *Xty
    #	in a least squares sense, subject to x >= 0.
    #	Differently stated it solves the problem min ||y - Xx|| if
    #	XtX = X'*X and Xty = X'*y.
    #
    #	A default tolerance of TOL = MAX(SIZE(XtX)) * NORM(XtX,1) * EPS
    #	is used for deciding when elements of x are less than zero.
    #	This can be overridden with x = fnnls(XtX,Xty,TOL).
    #
    #	[x,w] = fnnls(XtX,Xty) also returns dual vector w where
    #	w(i) < 0 where x(i) = 0 and w(i) = 0 where x(i) > 0.
    #
    #	See also NNLS and FNNLSb

    #	L. Shure 5-8-87
    #	Revised, 12-15-88,8-31-89 LS.
    #	(Partly) Copyright (c) 1984-94 by The MathWorks, Inc.

    #	Modified by R. Bro 5-7-96 according to
    #       Bro R., de Jong S., Journal of Chemometrics, 1997, 11, 393-401
    # 	Corresponds to the FNNLSa algorithm in the paper
    #
    #	Rasmus bro
    #	Chemometrics Group, Food Technology
    #	Dept. Dairy and Food Science
    #	Royal Vet. & Agricultural
    #	DK-1958 Frederiksberg C
    #	Denmark
    #	rb@...
    #	http://newton.foodsci.kvl.dk/users/rasmus.html
    #  Reference:
    #  Lawson and Hanson, "Solving Least Squares Problems", Prentice-
    Hall, 1974.
    #
    '''
    # initialize variables
    m = XtX.shape[0]
    n = XtX.shape[1]

    if tol == 0:
        eps = 2.2204e-16
        tol = 10 * eps * norm(XtX,1)*max(m, n);
        #end

    P = numpy.zeros(n, numpy.uint16)
    P[:] = -1
    Z = numpy.arange(0,n)

    z = numpy.zeros(m, numpy.float32)
    x = numpy.array(P)
    ZZ = numpy.array(Z)

    w = Xty - numpy.dot(XtX, x)

    # set up iteration criterion
    iter = 0
    itmax = 30 * n

    # outer loop to put variables into set to hold positive coefficients
    while any(Z) and any(w[ZZ] > tol) :
        print(max(w[ZZ]))
        wt = max(w[ZZ])
        t = find(w[ZZ] == wt)
        t = t[-1:][0]
        t = ZZ[t]
        P[t] = numpy.asarray(t, dtype=numpy.uint16)
        Z[t] = -1
        PP = find(P != -1)

        ZZ = find(Z != -1)
        if len(PP) == 1 :
            XtyPP = Xty[PP]
            XtXPP = XtX[PP, PP]
            z[PP] = XtyPP / XtXPP
        else :
            XtyPP = numpy.array(Xty[PP])
            XtXPP = numpy.array(XtX[PP, numpy.array(PP)[:,numpy.NewAxis]])
            z[PP] = numpy.dot(XtyPP, la.generalized_inverse(XtXPP))
            #end
        z[ZZ] = 0

    # inner loop to remove elements from the positive set which no longer belong
    while any(z[PP] <= tol) and (iter < itmax) :
        iter += 1
        iztol = find(z <= tol)
        ip = find(P[iztol] != -1)
        QQ = iztol[ip]
        if len(QQ) == 1 : alpha = x[QQ] / (x[QQ] - z[QQ])
        else :
            x_xz = x[QQ] / (x[QQ] - z[QQ])
            alpha = x_xz._lb()
            #end
        x += alpha * (z - x)
        iabs = find(abs(x) < tol)
        ip = find(P[iabs] != -1)
        ij = iabs[ip]
        Z[ij] = numpy.array(ij)
        P[ij] = -1
        PP = find(P != -1)
        ZZ = find(Z != -1)

        if len(PP) == 1 :
            XtyPP = Xty[PP]
            XtXPP = XtX[PP, PP]
            z[PP] = XtyPP / XtXPP
        else :
            XtyPP = numpy.array(Xty[PP])
            XtXPP = numpy.array(XtX[PP, numpy.array(PP)[:,numpy.NewAxis]])
            z[PP] = numpy.dot(XtyPP, la.generalized_inverse(XtXPP))
            #endif
        z[ZZ] = 0
        #end while
    x = numpy.array(z)
    w = Xty - numpy.dot(XtX, x)
    #end while
    return x, w
    #end def

if __name__ == '__main__' :
#
# fcs2 [x, w] = fnnls(Xt.X, Xt.y, tol)
# to solve min ||y - X.x|| s.t. x >= 0
#
# matlab:lsqnonneg
#   X = [1, 10, 4, 10; 4, 5, 1, 12; 5, 1, 9, 20];
#   y = [4; 7; 4]
#   x = lsqnonneg(X, y) => x = [0.9312; 0.3683; 0; 0];
#

    X = numpy.array([[1, 10, 4, 10], [4, 5, 1, 12], [5, 1, 9, 20]], numpy.float32)
    y = numpy.array([4, 7, 4], numpy.float32)

    if False :
        X = numpy.array([[1, 10, 4, 10],
                         [4, 5, 1, 12],
                         [5, 1, 9, 20],
                         [4, 3, 2, 1]], numpy.float32)
        y = numpy.array([4, 7, 4, 1], numpy.float32)
        #end

    if False :
        X = numpy.zeros((20, 20), numpy.float32)
        for n in range(20) : X[n,:] = numpy.arange(0.0, 400.0, step = 20)
        y = numpy.arange(0.0, 20.0)
        #end
    Xt = numpy.transpose(numpy.array(X))
    x, w = fnnls(numpy.dot(Xt, X), numpy.dot(Xt, y))

    print('X = ', X)
    print('y = ', y)
    print('x = ', x)
    #end if __name__ == '__main__'