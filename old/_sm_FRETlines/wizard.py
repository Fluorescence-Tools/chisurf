import numpy as np

model = cs.current_fit.model
r1 = cs.current_fit.model.parameter_dict['R(G,1)']
r2 = cs.current_fit.model.parameter_dict['R(G,2)']
r3 = cs.current_fit.model.parameter_dict['R(G,3)']

x1 = cs.current_fit.model.parameter_dict['x(G,1)']
x2 = cs.current_fit.model.parameter_dict['x(G,2)']
x3 = cs.current_fit.model.parameter_dict['x(G,3)']

s1 = cs.current_fit.model.parameters_all_dict['s(G,1)']
s2 = cs.current_fit.model.parameters_all_dict['s(G,2)']
s3 = cs.current_fit.model.parameters_all_dict['s(G,3)']

# # iso-line for a sinlge distance with width/distance
# x1.value = 1.0
# x2.value = 0.0
# x3.value = 0.0
# values = list()
# for rm in np.logspace(0.5, 2.5, 150):
#     for si in np.linspace(1., 30, 50):
#         r1.value = rm
#         s1.value = si
#         model.update()
#         eff = model.transfer_efficiency
#         tauf = model.fluorescence_averaged_lifetime
#         values.append([tauf, eff, rm, si])
#
# vls = np.array(values)
# np.savetxt('distance_width.tx4', vls, delimiter='\t')

# iso-line for a fraction distance with width/distance
r1.value = 30.0
r2.value = 50.0
r3.value = 70.0

s1.value = 1.
s2.value = 1.
s3.value = 1.

values = list()
n_points = 100
for x1i in np.linspace(0.0, 1.0, n_points):
    x2_3 = 1.0 - x1i
    x1.value = x1i
    for k3 in np.linspace(0.0, 1.0, n_points):
        x2.value = x2_3 * (1. - k3)
        x3.value = x2_3 * k3

        model.update()
        eff = model.fret_efficiency
        tauf = model.fluorescence_averaged_lifetime
        values.append([tauf, eff, x1.value, x2.value, x3.value])

vls = np.array(values)
np.savetxt('x1_x2_x3.txt', vls, delimiter='\t')


# Two-state limiting lines

x3i = 0.0  # x1<->x2
values = list()
for x1i in np.linspace(0.0, 1.0, 100):
    x2i = 1.0 - x1i

    x1.value = x1i
    x2.value = x2i
    x3.value = x3i
    model.update()
    eff = model.fret_efficiency
    tauf = model.fluorescence_averaged_lifetime
    values.append([tauf, eff, x1i, x2i, x3i])

vls = np.array(values)
np.savetxt('x1_x2.txt', vls, delimiter='\t')

x1i = 0.0  # x2<->x3
values = list()
for x2i in np.linspace(0.0, 1.0, 100):
    x3i = 1.0 - x2i

    x1.value = x1i
    x2.value = x2i
    x3.value = x3i
    model.update()
    eff = model.fret_efficiency
    tauf = model.fluorescence_averaged_lifetime
    values.append([tauf, eff, x1i, x2i, x3i])

vls = np.array(values)
np.savetxt('x2_x3.txt', vls, delimiter='\t')

x2i = 0.0  # x3<->x1
values = list()
for x3i in np.linspace(0.0, 1.0, 100):
    x1i = 1.0 - x3i

    x1.value = x1i
    x2.value = x2i
    x3.value = x3i
    model.update()
    eff = model.fret_efficiency
    tauf = model.fluorescence_averaged_lifetime
    values.append([tauf, eff, x1i, x2i, x3i])

vls = np.array(values)
np.savetxt('x3_x1.txt', vls, delimiter='\t')

# Static FRET-line
x1.value = 1.0
x2.value = 0.0
x3.value = 0.0

values = list()
for rda in np.logspace(0, 2.0, 100):
    r1.value = rda
    model.update()
    eff = model.fret_efficiency
    tauf = model.fluorescence_averaged_lifetime
    values.append([tauf, eff, rda])

vls = np.array(values)
np.savetxt('static_line.txt', vls, delimiter='\t')
