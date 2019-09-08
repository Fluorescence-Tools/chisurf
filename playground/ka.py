
def calc_vvdecay(fluorescence_lifetimes, t):
	vv = np.zeros_like(t)
#monoexponential lifetime decay
#	vv = (1-l1)*a1*np.exp(-t/tau1)+(1-l1)*2*a1*b1*np.exp(-t*(1/tau1+1/rho1))+l1*a1/gfactor*np.exp(-t/tau1)-l1*a1*b1/gfactor*np.exp(-t*(1/tau1+1/rho1))
#biexponential lifetime decay
	vv = ((1-l1)*a1*np.exp(-t/tau1)+(1-l1)*2*a1*b1*np.exp(-t*(1/tau1+1/rho1))+(1-l1)*a2*np.exp(-t/tau2)+(1-l1)*2*a2*b1*np.exp(-t*(1/tau2+1/rho1))+
        l1*a1/gfactor*np.exp(-t/tau1)-l1*a1*b1/gfactor*np.exp(-t*(1/tau1+1/rho1))+l1*a1/gfactor*np.exp(-t/tau2)-l1*a2*b1/gfactor*np.exp(-t*(1/tau2+1/rho1)))
	return vv

def calc_vhdecay(fluorescence_lifetimes, t):
	vh = np.zeros_like(t)
#monoexponential lifetime decay
#	vh = l2*a1*np.exp(-t/tau1)+l2*2*a1*b1*np.exp(-t*(1/tau1+1/rho1))+(1-l2)*a1/gfactor*np.exp(-t/tau1)-(1-l2)*a1*b1/gfactor*np.exp(-t*(1/tau1+1/rho1))
#biexponential lifetime decay
	vh = (l2*a1*np.exp(-t/tau1)+l2*2*a1*b1*np.exp(-t*(1/tau1+1/rho1))+l2*a2*np.exp(-t/tau2)+l2*2*a2*b1*np.exp(-t*(1/tau2+1/rho1))+
        (1-l2)*a1/gfactor*np.exp(-t/tau1)-(1-l2)*a1*b1/gfactor*np.exp(-t*(1/tau1+1/rho1))+(1-l2)*a2/gfactor*np.exp(-t/tau2)-(1-l2)*a2*b1/gfactor*np.exp(-t*(1/tau2+1/rho1)))
	return vh

def calc_anisotropy_decay(rotational_correlation_times, t):
	r = np.zeros_like(t)
#	for b, rho in rotational_correlation_times:
#		r += b * np.exp(-t/rho)
#	return r
	r += b1 * np.exp(-t/rho1)
	return r

# taux = a1*tau1
# tauf = (a1*tau1*tau1)/taux
taux = a1*tau1+a2*tau2
tauf = (a1*tau1*tau1+a2*tau2*tau2)/taux
rss = r0/(1+tauf/rho1)
nr_photons_vv = gfactor*nr_photons_vh*((2*rss+1)/(1-rss))

# calculate vh decay
vh_y = calc_vhdecay(fluorescence_lifetimes, t)
vh_y_conv = np.convolve(irf_count_vh, vh_y, mode='full')[:len(vh_y)]
vh_y_conv *= nr_photons_vh / sum(vh_y_conv)

vh_y_noise = np.random.poisson(vh_y_conv)
p.semilogy(t, irf_count_vh)
p.semilogy(t, vh_y_noise)
p.semilogy(t, vh_y_conv)
p.show()

# calculate vv decay
vv_y = calc_vvdecay(fluorescence_lifetimes, t)
vv_y_conv = np.convolve(irf_count_vv, vv_y, mode='full')[:len(vv_y)]
vv_y_conv *= nr_photons_vv / sum(vv_y_conv)

vv_y_noise = np.random.poisson(vv_y_conv)
p.semilogy(t, irf_count_vv)
p.semilogy(t, vv_y_noise)
p.semilogy(t, vv_y_conv)
p.show()

# anisotropy based on input parameter
r_y = calc_anisotropy_decay(rotational_correlation_times, t)

# anisotropy based on simulated data
aniso_calc = (vv_y_noise - gfactor * vh_y_noise)/(vv_y_noise + 2 * gfactor * vh_y_noise)
aniso_fit = (vv_y_conv - gfactor * vh_y_conv)/(vv_y_conv + 2 * gfactor * vh_y_conv)
p.plot(t, aniso_calc)
p.plot(t, aniso_fit)
p.plot(t, r_y)
p.show()

rsscalc = (sum(vv_y_conv) - gfactor*sum(vh_y_conv))/(sum(vv_y_conv) + 2*gfactor*sum(vh_y_conv))

variables = [nr_photons_vh, nr_photons_vv, gfactor, l1, l2, a1, tau1, a2, tau2, taux, tauf, r0, b1, rho1, rss, rsscalc]
#variables_str = join(str(x) for x in variables
variables_name = ['Photons VH', 'Photons_VV', 'gfactor', 'L1', 'L2', 'a1', 'lt1', 'a22', 'lt2', 'tx', 'tf', 'r0', 'b1', 'rho1', 'rss', 'rss_calc']

print(variables_name, variables)

#output = np.vstack([variables_name, variables_str])
#np.savetxt('Z:\Katherina Hemmen\Skripts\Generate_decay\Input.txt', output, delimiter=' ')

np.savetxt('D:\Katherina\Aniso_sim\Decay_VV.txt', np.vstack([t, vv_y_noise]).T, delimiter=' ')
np.savetxt('D:\Katherina\Aniso_sim\Decay_VH.txt', np.vstack([t, vh_y_noise]).T, delimiter=' ')
np.savetxt('D:\Katherina\Aniso_sim\Anisotropy.txt', np.vstack([t, aniso_calc]).T, delimiter=' ')
