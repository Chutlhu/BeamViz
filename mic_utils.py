import numpy as np
import pyroomacoustics as pra

from pyroomacoustics.directivities import (
	DirectivityPattern,
	DirectionVector,
	CardioidFamily,
	cardioid_func
)
import geo_utils as geo


def get_easycom_array(center=None, binaural_mic=True):
	if center is None:
		center = np.zeros([3,1])
	center = geo.check_geo_dim_convention(center)

	# Channel/Mic 	# 	X (mm) 	Y (mm) 	Z (mm)
	# 				1 	 82		-5 		-29
	# 				2 	-01		-1	 	 30
	#				3 	-77 	-2 		 11
	# 				4 	-83 	-5 		-60
	# 				5	N/A 	N/A 	N/A
	# 				6	N/A 	N/A 	N/A
	R_3M = np.array([
		[ 0.082, -0.005, -0.029],
		[-0.001, -0.001,  0.030],
		[-0.077, -0.002,  0.011],
		[-0.083, -0.005, -0.060],
		# [ 0.052, -0.010, -0.060],
		# [-0.053, -0.005, -0.060],
	]).T
	# use my usual convention (left hand)
	R_3M = R_3M[[2,0,1],:]
	# R_3M = R_3M[[1,0,2],:]
	# R_3M[1,:] *= -1
	
	if binaural_mic: 
		n_mics = 6
	else:
		n_mics = 4
		R_3M = R_3M[:,:4]
 
	P_3M = R_3M + center
		
	geo.check_geo_dim_convention(P_3M)

	return P_3M


def get_array(array_type, array_center, array_prop=None):

	geo.check_geo_dim_convention(array_center)

	if array_type == 'circular':
		radius = array_prop['spacing']
		phi = array_prop['phi']
		n_mics = array_prop['n_mics']
		mic_pos = pra.circular_2D_array(center=array_center[:2,0], M=n_mics, phi0=phi, radius=radius)
		mic_pos = np.concatenate([mic_pos, np.zeros([1,n_mics])], axis=0)

	elif array_type == 'linear':
		spacing = array_prop['spacing']
		phi = array_prop['phi']
		n_mics = array_prop['n_mics']
		mic_pos = pra.linear_2D_array(center=array_center[:2,0], M=n_mics, phi=np.deg2rad(phi), d=spacing)
		mic_pos = np.concatenate([mic_pos, np.zeros([1,n_mics])], axis=0)
	elif array_type == 'easycom':
		mic_pos = get_easycom_array(center=None)
		n_mics = mic_pos.shape[1]
	else:
		raise ValueError('Select correct array')
		
	geo.check_geo_dim_convention(mic_pos)

	return mic_pos
	

# def get_directivity(pattern, directions, ):
# 	assert mic_pos.shape == directions.shape
		
# 	pattern = DirectivityPattern[pattern]
# 	# compute response
# 	resp = cardioid_func(
# 					x=mic_pos[:,0].T, direction=directions.unit_vector,
# 					coef=pattern.value, 
# 					magnitude=True)
# 	print(resp)
# 	1/0
# 	return resp