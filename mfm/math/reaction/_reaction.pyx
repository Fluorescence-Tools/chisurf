#from numpy import arange, array, empty,zeros
import time

cimport numpy as np
import numpy as np
from numpy.random import multinomial

##### SSA #########
# http://pyinsci.blogspot.de/2008/10/fast-gillespies-direct-algorithm-in.html

DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef np.int_t INT_t

cdef class Model:
    cdef object vn,rates,inits,pv
    cdef np.ndarray tm,res,time,series
    cdef int pvl,nvars,steps
    cdef object ts
    def __init__(self,vnames,rates,inits, tmat,propensity):
        '''
         * vnames: list of strings
         * rates: list of fixed rate parameters
         * inits: list of initial values of variables
         * propensity: list of lambda functions of the form:
            lambda r,ini: some function of rates ans inits.
        '''
        self.vn = vnames
        self.rates = rates
        self.inits = inits
        self.tm = tmat
        self.pv = propensity#[compile(eq,'errmsg','eval') for eq in propensity]
        self.pvl = len(self.pv) #length of propensity vector
        self.nvars = len(self.inits) #number of variables
        self.time = np.zeros(1)
        self.series = np.zeros(1)
        self.steps = 0

    def run(self, method='SSA', int tmax=10, int reps=1):
        cdef np.ndarray[DTYPE_t,ndim=3] res = np.zeros((tmax,self.nvars,reps),dtype=float)
        tvec = np.arange(tmax)
        self.res = res
        cdef int i, steps
        if method =='SSA':
            for i from 0 <= i<reps:
                steps = self.GSSA(tmax,i)
            print(steps,' steps')
        elif method == 'SSAct':
            pass
        self.time=tvec
        self.series=self.res
        self.steps=steps

    def getStats(self):
        return self.time,self.series,self.steps

    cpdef int GSSA(self, int tmax=50,int round=0):
        '''
        Gillespie Direct algorithm
        '''
        ini = self.inits
        r = self.rates
        pvi = self.pv
        cdef int l,steps,i,tim
        cdef double a0,tc, tau
        #cdef np.ndarray[INT_t] tvec
        cdef np.ndarray[DTYPE_t] pv
        l=self.pvl
        pv = np.zeros(l, dtype=float)
        tm = self.tm
        #tvec = np.arange(tmax,dtype=int)
        tc = 0
        steps = 0
        self.res[0,:,round]= ini
        a0=1.
        for tim from 1<= tim <tmax:
            while tc < tim:
                for i from 0 <= i <l:
                    pv[i] = pvi[i](r,ini)
                #pv = abs(array([eq() for eq in pvi]))# #propensity vector
                a0 = a_sum(pv,l) #sum of all transition probabilities
                #print ini#,tim, pv, a0
                tau = (-1/a0)*np.log(np.random.random())
                event = multinomial(1,(pv/a0)) # event which will happen on this iteration
                ini += tm[:,event.nonzero()[0][0]]
                #print tc, ini
                tc += tau
                steps +=1
                if a0 == 0: break
            self.res[tim,:,round] = ini
            if a0 == 0: break
#        tvec = tvec[:tim]
#        self.res = res[:tim,:,round]
        return steps

    def CR(self,pv):
        """
        Composition reaction algorithm
        """
        pass


def main():
    vars = ['s','i','r']
    cdef np.ndarray ini= np.array([500,1,0],dtype = int)
    cdef np.ndarray rates = np.array([.001,.1],dtype=float)
    cdef np.ndarray tm = np.array([[-1,0],[1,-1],[0,1]])

    prop = [l1,l2]
    M = Model(vnames = vars,rates = rates,inits=ini, tmat=tm,propensity=prop)
    t0=time.time()
    M.run(tmax=80,reps=1000)
    print('total time: ' % time.time()-t0)


cdef double a_sum(np.ndarray a, int len):
    cdef double s
    cdef int i
    s=0
    for i from 0 <=i <len:
        s+=a[i]
    return s

def l1(np.ndarray r,np.ndarray ini):
    return r[0]*ini[0]*ini[1]
def l2(np.ndarray r,np.ndarray ini):
    return r[1]*ini[1]
