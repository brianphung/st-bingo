from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from research.utility.rotations import align_pi_plane_with_axes_rot
from research.utility.rotations import align_axes_with_pi_plane_rot
from plot_mapping import draw_principal_axes
from numpy import sin, cos

def mapping_model(eps):

    X_0 = eps
    if not isinstance(eps, np.ndarray):
        eps = np.array([eps])[:, None, None]
    
    alpha = 180*np.pi/180
    eq = np.array([ [1,0,0], [0, cos(alpha), -sin(alpha)], [ 0, sin(alpha), cos(alpha) ] ])

    if eq.shape[0] != eps.shape[0]:
        eq = np.repeat(eq[None, :, :], eps.shape[0], axis=0)
    return eq
# IMPORT THE MAPPING MODEL HERE
#from plot_mapping import vm_1_mapping_model as mapping_model

Y0 = 10
H = 990
# Generate a stress-strain curve
number_of_eqps_values = 21

eqps = np.array([ i*0.01 for i in range(number_of_eqps_values) ])
zeros = np.ones(number_of_eqps_values)*(100*eqps)
stress_X = [ Y0 + H*eqps_ for eqps_ in eqps ]
stress_states = np.column_stack([stress_X, zeros, zeros])

mapping_matrices = np.array([ mapping_model(eqps_) for eqps_ in eqps ])
original_trajectory = (stress_states @ align_axes_with_pi_plane_rot())[:, :2]


mapping_stresses = np.array([ (mapping_matrices[i] @ stress_states[i])[0] for i in range(number_of_eqps_values) ])
#print(mapping_matrices[0] @ stress_states[0])
mapping_trajectory = (mapping_stresses @ align_axes_with_pi_plane_rot())[:, :2]

# Unmap those stresses
inverse_mapping_matrices = np.array([ np.linalg.inv(mapping_model(eqps_)) for eqps_ in eqps ])
inv_mapping_stresses = np.array([ (inverse_mapping_matrices[i] @ mapping_stresses[i])[0] for i in range(number_of_eqps_values) ])
inverse_mapping_trajectory = (inv_mapping_stresses @ align_axes_with_pi_plane_rot())[:, :2]
#print(mapping_trajectory)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(*original_trajectory.T, marker='*', label="real")
ax.scatter(*mapping_trajectory.T, s=20, marker='o', label="fictious")
ax.scatter(*inverse_mapping_trajectory.T, marker='.', s= 2, c='r', label="real (mapped and remapped)")
draw_principal_axes(ax, length_of_axes=200, scale=15)


ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])

plt.show()