import numpy as np
from PowDDeR_plots import *

# Notes:
#	- Need to add try/except
#	- Connection adaptive capacity in progress and needs tested

class Distribution_network:
	'''Adaptive capacity of the network configuration'''

	number_of_buses  = 0
	number_of_assets = 0
	time_minutes = None
	pfa_radians  = None

	# Constructor
	def __init__(self, name='network'):
		self.name = name
		self.ASRs = [];
		self.list_of_ASRs = []

	# Add an ASR to the network
	def add_ASR(self, ASR):
		self.ASRs.append(ASR)

	# Remove an ASR to the network
	def remove_ASR(self, ASR):
		self.ASRs.remove(ASR)

	# Clears the network object
	def clear_network():
		self.ASRs = []

	# Print contenets of the network
	def __str__(self):
		return_str = self.name + ':\n'
		
		for ASR in self.ASRs:
			return_str += '\t' + ASR.name + ': ['
			
			for bus in ASR.list_of_busses:
				return_str += bus.name + ', '
	
			return_str += "]"
			return_str = return_str.replace(", ]", "]")
			return_str += "\n"
			
		return return_str

class ASR:
	"""The Aggregated System Resources keeps the aggregated adaptive capacity of all the Buses in the ASR"""

	number_of_ASRs = 0
	time_minutes = None
	pfa_radians = None

	# Constructor
	def __init__(self, name='ASR'):
		self.name = name
		self.number_of_buses = 0
		self.number_of_assets = 0
		self.list_of_busses = []
		self.list_of_assets = []
		self.connected_to_ASRs = []
		ASR.number_of_ASRs += 1

	# Add a bus to the ASR
	def add_bus_to_ASR(self, bus):

		# If it is the first bus or asset added keep track for plotting
		if self.number_of_assets == 0:
			ASR.time_minutes = bus.time_minutes
			ASR.pfa_radians = bus.pfa_radians
		self.number_of_buses += 1
		self.number_of_assets += bus.number_of_assets
		self.list_of_busses.append(bus)
		for asset in bus.list_of_assets:
			self.list_of_assets.append(asset)

	# Remove a bus from the ASR
	def remove_bus_from_ASR(self, bus):
		self.number_of_buses -= 1
		for asset in bus.list_of_assets:
			self.list_of_assets.remove(asset)

	# Add an asset to the ASR
	def add_asset_to_ASR(self, asset):
		if self.number_of_assets == 0:
			ASR.time_minutes = asset.time_minutes
			ASR.pfa_radians = asset.pfa_radians
		self.number_of_assets += 1
		self.list_of_assets.append(asset)

	# Remove asset from the ASR
	def remove_asset_from_ASR(self, asset):
		self.num_of_assets -= 1
		self.list_of_assets.remove(asset)

	# Add connection 
	def add_connection(self, ASR, line):
		self.connected_to_ASRs.append((ASR, line))
		ASR.connected_from_ASRs.append((self, line))

	# Remove connection
	def remove_connection(self, ASR):
		index = [x[0] for x in self.connected_to_ASRs].index(ASR)
		self.connected_to_ASRs.pop(index)
		index = [x[0] for x in self.connected_from_ASRs].index(ASR)
		self.connected_from_ASRs.pop(index)

	# Clear everything from the ASR
	def clear_ASR(self):
		self.num_of_buses = 0
		self.number_of_assets = 0
		self.list_of_busses = []
		self.list_of_assets = []
		self.connected_to_ASRs = []

	# Print ASR assets
	def __str__(self):
		ASR_bus_asset_str = self.name + ':\n'
		
		for bus in self.list_of_busses:
			ASR_bus_asset_str += '\t' + bus.name + ': ['
			
			for asset in bus.list_of_assets:
				ASR_bus_asset_str += asset.name + ', '
	
			ASR_bus_asset_str += "]"
			ASR_bus_asset_str = ASR_bus_asset_str.replace(", ]", "]")
			ASR_bus_asset_str += "\n"
			
		return ASR_bus_asset_str

class Bus:
	""" The Bus keeps the aggregated adaptive capacity of all the Assets in the Bus"""
	
	number_of_buses  = 0
	time_minutes = None
	pfa_radians  = None
	i = None

	# Constructor for bus object
	def __init__(self, name="Bus"):
		self.name = name
		self.list_of_assets = []
		self.number_of_assets = 0
		self.connected_to_buses = []
		self.connected_from_buses = []
		Bus.number_of_buses += 1


	# Add an asset to the bus
	def add_asset_to_bus(self, asset):
		if self.number_of_assets == 0:
			Bus.time_minutes = asset.time_minutes
			Bus.pfa_radians = asset.pfa_radians
			Bus.i = asset.i

		self.number_of_assets += 1
		self.list_of_assets.append(asset)

	# Remove an asset from the bus
	def remove_asset_from_bus(self, asset):
		self.number_of_assets -= 1
		self.list_of_assets.remove(asset)


	# Add bus connection 
	def add_bus_connected_to(self, bus, line):
		# Need to connect from lower bus number to higher bus number. 
		# Power flow in line is assumed to go this direction
		self.connected_to_buses.append((bus, line))
		bus.connected_from_buses.append((self, line))

	# Remove bus connection
	def remove_bus_connected_to(self, bus):
		index = [x[0] for x in self.connected_to_buses].index(bus)
		self.connected_to_buses.pop(index)
		index = [x[0] for x in self.connected_from_buses].index(bus)
		self.connected_from_buses.pop(index)

	# Clear everything from bus
	def clear_bus(self):
		self.num_of_assets = 0
		self.list_of_assets = []
		self.connected_to_buses = []
		self.connected_from_buses = []

	# Print the assets and connections of the bus
	def __str__(self):

		return_str = self.name + ':\n\tAssets: '
		
		# Add the list of assets
		if not self.list_of_assets:
			return_str += '\n\tConnected to: '

		for i, asset in enumerate(self.list_of_assets):
			if i == self.number_of_assets-1:
				return_str += asset.name + '\n\tConnected to: '
			else:
				return_str += asset.name + ', '

		n = len(self.connected_to_buses)
		for i, bus in enumerate(self.connected_to_buses):
			if i == n-1:
				return_str += bus[0].name + '\n\tConnected from: '
			else:
				return_str += bus[0].name + ', '

		n = len(self.connected_from_buses)
		for i, bus in enumerate(self.connected_from_buses):
			if i == n-1:
				return_str += bus[0].name
			else:
				return_str += bus[0].name + ', '

		return return_str

class Asset:
	"""Asset is the parent object for all assets"""

	total_assets      = 0		# Keeps the total number of assets
	max_time_minutes  = 10		# Max time for manifold (min) 
	number_radial_pts = 90		# Number of radial points per quadrant

	# Index of different quadrants
	i = {'q1': np.arange(                  0,   number_radial_pts),
	     'q2': np.arange(  number_radial_pts, 2*number_radial_pts),
	     'q3': np.arange(2*number_radial_pts, 3*number_radial_pts),
	     'q4': np.arange(3*number_radial_pts, 4*number_radial_pts)};

	# Time arrays over 3 days with varying time steps
	t1 = np.arange(      0,      60,    1).reshape(-1,1)	# First minute in 1 second time steps
	t2 = np.arange(     60,     180,    5).reshape(-1,1)	# From 1 minute to 3 minutes, 5 second time steps
	t3 = np.arange(    180,     600,   15).reshape(-1,1)  	# From 3 minute to 10 minute, 15 second time steps
	t4 = np.arange(    600,    3600,  120).reshape(-1,1)	# From 10 minutes to 1 hour, 2 minute time steps
	t5 = np.arange(   3600, 24*3600,  600).reshape(-1,1)	# From 1 hour to 1 day, 10 minute time steps
	t6 = np.arange(24*3600, 73*3600, 1200).reshape(-1,1)	# From 1 day to 3 days, 20 minute time steps (73 hours because exclusive function)

	# Concatenate and trim array at max time 
	t_s = np.concatenate( (t1, t2, t3, t4, t5, t6), axis = 0)
	idx_t_end = np.argmax(t_s > max_time_minutes * 60)

	# Index of maximum time
	if max_time_minutes < 60*24*3:
		time_seconds = t_s[0:idx_t_end]
	else:
		time_seconds = t_s
		if (max_time_minutes * 60) > t_s[-1]:
			print('Max time was too large, reduced to 3 days')

	# Time matrix used for plotting [min]
	time_minutes = np.repeat(time_seconds, number_radial_pts * 4, axis=1) / 60

	# Power Factor Angles (pfa) used in adaptive capacity calcs
	pfa_deg = np.linspace(0.5 * (90 / number_radial_pts), 360 - 0.5 * (90 / number_radial_pts), number_radial_pts * 4)
	pfa_radians = np.deg2rad(pfa_deg)
	sin_pfa = np.sin(pfa_radians)
	cos_pfa = np.cos(pfa_radians)

	# Constructor for the asset objects
	def __init__(self, 	
				 name, 
				 P_output, 
				 Q_output, 
				 P_nameplate_pos_neg, 
				 Q_nameplate_pos_neg, 
				 latency, 
				 P_time_ramp_up, 
				 P_time_ramp_down, 
				 Q_time_ramp_up, 
				 Q_time_ramp_down,
				 P_real_time_max=None,
				 uncertainty=None, 
				 inertia=None):
		
		# Set the object values
		self.name     = name
		self.P_output = P_output
		self.Q_output = Q_output
		self.P_max    = P_nameplate_pos_neg[0]
		self.P_min    = P_nameplate_pos_neg[1]
		self.Q_max    = Q_nameplate_pos_neg[0]
		self.Q_min    = Q_nameplate_pos_neg[1]
		self.latency  = latency
		self.P_real_time_max = P_real_time_max
		self.uncertainty     = uncertainty
		self.inertia         = inertia

		# If there is a real time maximum, i.e solar, wind, hydro, output can't be larger and is reduced to actual max
		if self.P_real_time_max is not None and self.P_output > self.P_real_time_max:
			self.P_output = self.P_real_time_max
			print(self.name + ' power output was above capability, reduced to ' + str(self.P_output))

		if self.P_output > self.P_max:
			self.P_output = self.P_max
			print(f'The {self.name} P output was reduced to {self.P_max}, its maximum output capability')

		# Line assets do not have any ramp time
		if isinstance(self, Line):
			self.d_dt_ramp = {'P_up':0, 'P_down':0, 'Q_up':0, 'Q_down':0}
		else:
			self.d_dt_ramp = {'P_up':   (self.P_max - self.P_min) / P_time_ramp_up, 
			                  'P_down':-(self.P_max - self.P_min) / P_time_ramp_down,
			                  'Q_up':   (self.Q_max - self.Q_min) / Q_time_ramp_up,
			                  'Q_down':-(self.Q_max - self.Q_min) / Q_time_ramp_down}

		# Increment the number of assets
		Asset.total_assets += 1

		# Only call asset_domain once, it doesn't change
		self.asset_domain()
		self.update_asset_adaptive_capacity()


	def update_asset_adaptive_capacity(self):
		"""Update the adaptive capacity of an asset. Call if state of asset has changed"""

		self.calc_flexibility()
		self.calc_temporal_power_arrays()
		self.calc_adaptive_capacity()
		self.calc_energy_limits()

	
	def asset_domain(self):
		"""Defines the bounding domain of the asset in cartesian coordinates (P and Q)"""

		# First define the power factor angle that will be used
		# Domain uses a lot of points, they are not used in the adaptive capacity calcs
		n_points_per_quad = 180				
		start   = 90 / (n_points_per_quad * 2)
		pfa     = np.deg2rad(np.arange(start, 360+start, start*2))
		sin_pfa = np.sin(pfa)
		cos_pfa = np.cos(pfa)
		
		# Index of each quadrant (different from i)
		i_q1 = np.arange(                  0,   n_points_per_quad)
		i_q2 = np.arange(  n_points_per_quad, 2*n_points_per_quad)
		i_q3 = np.arange(2*n_points_per_quad, 3*n_points_per_quad)
		i_q4 = np.arange(3*n_points_per_quad, 4*n_points_per_quad)

		# Distance to the real boundary at different pfa's
		S_P = np.empty_like(pfa)
		S_P[i_q1] = cos_pfa[i_q1] * self.P_max
		S_P[i_q2] = cos_pfa[i_q2] * self.P_min
		S_P[i_q3] = cos_pfa[i_q3] * self.P_min
		S_P[i_q4] = cos_pfa[i_q4] * self.P_max

		# Distance to the reactive boundary at pfa's
		S_Q = np.empty_like(pfa)
		S_Q[i_q1] = sin_pfa[i_q1] * self.Q_max
		S_Q[i_q2] = sin_pfa[i_q2] * self.Q_max
		S_Q[i_q3] = sin_pfa[i_q3] * self.Q_min
		S_Q[i_q4] = sin_pfa[i_q4] * self.Q_min
        
		# Distance from (0, 0) to the domain boundary at pfa's
		S_limit = np.sqrt(np.square(S_P) + np.square(S_Q))

		# The real power limit in each quadrant
		P = S_limit * cos_pfa
		P_domain       =  np.empty_like(P)
		P_domain[i_q1] =  np.minimum( P[i_q1],  self.P_max)
		P_domain[i_q2] = -np.minimum(-P[i_q2], -self.P_min)
		P_domain[i_q3] = -np.minimum(-P[i_q3], -self.P_min)
		P_domain[i_q4] =  np.minimum( P[i_q4],  self.P_max)

		# The reactive power limit in each quadrant
		Q = S_limit * sin_pfa      
		Q_domain = np.empty_like(P_domain)
		Q_domain[i_q1] =  np.minimum( Q[i_q1],  self.Q_max)
		Q_domain[i_q2] =  np.minimum( Q[i_q2],  self.Q_max)
		Q_domain[i_q3] = -np.minimum(-Q[i_q3], -self.Q_min)
		Q_domain[i_q4] = -np.minimum(-Q[i_q4], -self.Q_min)

		# Dictionary containing the domain limit of the asset
		self.domain = {'P':P_domain, 'Q':Q_domain}


	def calc_flexibility(self):
		"""Calculate the flexibility of the asset from the current operating point"""

		# Distance from operating point to the domain limit
		P = self.domain.get('P') - self.P_output
		Q = self.domain.get('Q') - self.Q_output

		# Find the angle from the operating point to domain points
		# This allows the assets to be aggregated
		angle = np.arctan2(Q, P)
		angle = np.where(angle < 0, angle + 2*np.pi, angle)
		angle_sort, P_sort = zip(*sorted(zip(angle, P)))
		angle_sort, Q_sort = zip(*sorted(zip(angle, Q)))

		# Interpolate at the new power factor angles
		P_flex_2D = np.interp(Asset.pfa_radians, angle_sort, P_sort)
		Q_flex_2D = np.interp(Asset.pfa_radians, angle_sort, Q_sort)
		S_flex_2D = np.sqrt(np.square(P_flex_2D)  + np.square(Q_flex_2D))

		# Dictinary containing the flexibility in cartesion (P,Q) and radial (S) 
		self.flex = {'Uncertainty':False, 'P':P_flex_2D, 'Q':Q_flex_2D, 'S':S_flex_2D, 'P_pos_unc':0, 'P_neg_unc':0}

		# If there is no uncertainty in the assets capability
		if self.uncertainty is None:

			# Flexibility based on real time max, i.e. solar, wind, etc...
			if self.P_real_time_max is not None:
				P_flex_2D = np.minimum(self.flex.get('P'), (self.P_real_time_max - self.P_output))
				self.flex.update({'P':P_flex_2D})
		
		# There is uncertainty in the asset, keep track of positive and negative capability in real power
		# Might need to add check in here to ensure P_output doesn't exceed P_maximum when negative uncertinty case
		else:
			uncertainty = self.P_output * self.uncertainty/100
			P_flex_2D_pos_unc = np.minimum(P_flex_2D, np.minimum((self.P_real_time_max - self.P_output + uncertainty), (self.P_max - self.P_output)))
			P_flex_2D = np.minimum(P_flex_2D, (self.P_real_time_max - self.P_output))
			P_flex_2D_neg_unc = np.minimum(P_flex_2D, (self.P_real_time_max - self.P_output - uncertainty))
			self.flex.update({'Uncertainty':True, 'P':P_flex_2D, 'P_pos_unc':P_flex_2D_pos_unc, 'P_neg_unc':P_flex_2D_neg_unc})


	def calc_temporal_power_arrays(self):
		"""Temporal constraints of asses; latency and ramp rate"""

		# Get the index of the latency
		if self.P_output != 0:
			i_latency = 0
		else:
			i_latency = np.argmax(self.time_seconds >= self.latency)
		
		# If latency is 0 because it is running (may need to change this for certain assets that have long latency even when running)
		if i_latency == 0:
			# Power increasing, real power quads 1&4, reactive power quads 1&2
			Pt_q_1_4 = np.multiply(self.time_seconds, self.d_dt_ramp.get('P_up'))
			Qt_q_1_2 = self.time_seconds * self.d_dt_ramp.get('Q_up')

			# Power decreasing, real power quads 2&3, reactive power quads 3&4		
			Pt_q_2_3 = self.time_seconds * self.d_dt_ramp.get('P_down')
			Qt_q_3_4 = self.time_seconds * self.d_dt_ramp.get('Q_down')
	
		# If latency is not 0, i.e. not running	
		else: 			
			# Zeros before latency is met
			zeros = np.zeros(i_latency).reshape(-1, 1)

			time_adj = np.maximum(self.time_seconds - self.latency, 0)
			Pt_q_1_4 = np.multiply(time_adj, self.d_dt_ramp.get('P_up'))
			Qt_q_1_2 = np.multiply(time_adj, self.d_dt_ramp.get('Q_up'))

			Pt_q_2_3 = np.multiply(time_adj, self.d_dt_ramp.get('P_down'))
			Qt_q_3_4 = np.multiply(time_adj, self.d_dt_ramp.get('Q_down'))

		# These are the arrays from the operating point at 0, pi/2, pi, 3pi/4
		self.ramp = {'P_up':   Pt_q_1_4.reshape(-1,1),
		             'Q_up':   Qt_q_1_2.reshape(-1,1),
		             'P_down': Pt_q_2_3.reshape(-1,1),
		             'Q_down': Qt_q_3_4.reshape(-1,1)}


	def calc_adaptive_capacity(self):
		"""Adaptive capacity of an asset"""

		# The 3D domain bound based on the flexibility only
		S_flex_3D = np.repeat(self.flex.get('S').reshape(-1,1), self.time_seconds.size, axis=1).T
		P_flex_3D = np.repeat(self.flex.get('P').reshape(-1,1), self.time_seconds.size, axis=1).T
		Q_flex_3D = np.repeat(self.flex.get('Q').reshape(-1,1), self.time_seconds.size, axis=1).T

		# These will be the domain bound based on the temporal ramps only (not bound by the asset output limits)
		S_temporal_3D = np.empty_like(P_flex_3D)

		S_temporal_3D[:, Asset.i.get('q1')] = np.minimum(self.ramp.get('P_up')   / Asset.cos_pfa[Asset.i.get('q1')], self.ramp.get('Q_up')   / Asset.sin_pfa[Asset.i.get('q1')])
		S_temporal_3D[:, Asset.i.get('q2')] = np.minimum(self.ramp.get('P_down') / Asset.cos_pfa[Asset.i.get('q2')], self.ramp.get('Q_up')   / Asset.sin_pfa[Asset.i.get('q2')])
		S_temporal_3D[:, Asset.i.get('q3')] = np.minimum(self.ramp.get('P_down') / Asset.cos_pfa[Asset.i.get('q3')], self.ramp.get('Q_down') / Asset.sin_pfa[Asset.i.get('q3')])
		S_temporal_3D[:, Asset.i.get('q4')] = np.minimum(self.ramp.get('P_up')   / Asset.cos_pfa[Asset.i.get('q4')], self.ramp.get('Q_down') / Asset.sin_pfa[Asset.i.get('q4')])
		
		P_temporal_3D = S_temporal_3D * np.cos(self.pfa_radians)

		S_adaptive_capacity_3D = np.minimum(S_flex_3D, S_temporal_3D)
		Q_ac_3D = np.sin(self.pfa_radians) * S_adaptive_capacity_3D

		# The adaptive capacity is the minimum between the flexibility domain and the temporal domain
		P_ac_3D = np.empty_like(P_flex_3D)
		P_ac_3D[:, Asset.i.get('q1')] =  np.minimum(       P_flex_3D[:, Asset.i.get('q1')],         P_temporal_3D[:, Asset.i.get('q1')])	
		P_ac_3D[:, Asset.i.get('q2')] = -np.minimum(np.abs(P_flex_3D[:, Asset.i.get('q2')]), np.abs(P_temporal_3D[:, Asset.i.get('q2')]))
		P_ac_3D[:, Asset.i.get('q3')] = -np.minimum(np.abs(P_flex_3D[:, Asset.i.get('q3')]), np.abs(P_temporal_3D[:, Asset.i.get('q3')]))
		P_ac_3D[:, Asset.i.get('q4')] =  np.minimum(       P_flex_3D[:, Asset.i.get('q4')],         P_temporal_3D[:, Asset.i.get('q4')])

		# Dictinary of the adaptive capacity of the asset
		self.ac = {'Uncertainty':False, 'P':P_ac_3D, 'Q':Q_ac_3D, 'P_pos_unc':0, 'P_neg_unc':0}
		
		# If there is uncertainty in the asset
		if self.uncertainty is not None:
			self.adaptive_capacity_with_uncertainty(P_temporal_3D)

	def adaptive_capacity_with_uncertainty(self, P_temporal_3D):
		# First do the positive uncertainty
		P_flex_3D_pos_unc = np.repeat(self.flex.get('P_pos_unc').reshape(-1,1), self.time_seconds.size, axis=1).T
		P_ac_pos_unc = np.empty_like(self.ac.get('P'))
		P_ac_pos_unc[:, Asset.i.get('q1')] =  np.minimum(       P_flex_3D_pos_unc[:, Asset.i.get('q1')],         P_temporal_3D[:, Asset.i.get('q1')])	
		P_ac_pos_unc[:, Asset.i.get('q2')] = -np.minimum(np.abs(P_flex_3D_pos_unc[:, Asset.i.get('q2')]), np.abs(P_temporal_3D[:, Asset.i.get('q2')]))
		P_ac_pos_unc[:, Asset.i.get('q3')] = -np.minimum(np.abs(P_flex_3D_pos_unc[:, Asset.i.get('q3')]), np.abs(P_temporal_3D[:, Asset.i.get('q3')]))
		P_ac_pos_unc[:, Asset.i.get('q4')] =  np.minimum(       P_flex_3D_pos_unc[:, Asset.i.get('q4')],         P_temporal_3D[:, Asset.i.get('q4')])

		# Next do the negative uncertainty
		if self.flex.get('P_neg_unc')[0] < 0:
			self.flex.update({'P_neg_unc': self.flex.get('P_neg_unc') - self.flex.get('P_neg_unc')[0] })

		P_flex_3D_neg_unc = np.repeat(self.flex.get('P_neg_unc').reshape(-1,1), self.time_seconds.size, axis=1).T
		P_ac_neg_unc = np.empty_like(self.ac.get('P'))
		P_ac_neg_unc[:, Asset.i.get('q1')] =  np.minimum(       P_flex_3D_neg_unc[:, Asset.i.get('q1')],         P_temporal_3D[:, Asset.i.get('q1')])
		P_ac_neg_unc[:, Asset.i.get('q2')] = -np.minimum(np.abs(P_flex_3D_neg_unc[:, Asset.i.get('q2')]), np.abs(P_temporal_3D[:, Asset.i.get('q2')]))
		P_ac_neg_unc[:, Asset.i.get('q3')] = -np.minimum(np.abs(P_flex_3D_neg_unc[:, Asset.i.get('q3')]), np.abs(P_temporal_3D[:, Asset.i.get('q3')]))
		P_ac_neg_unc[:, Asset.i.get('q4')] =  np.minimum(       P_flex_3D_neg_unc[:, Asset.i.get('q4')],         P_temporal_3D[:, Asset.i.get('q4')])

		self.ac.update({'Uncertainty':True, 'P_pos_unc':P_ac_pos_unc, 'P_neg_unc':P_ac_neg_unc})

	
	def calc_energy_limits(self):
		"""Energy constraint of assets"""

		# Energy limits are defined in each asset
		return

	def __str__(self):

		return_str = self.name + ':\n\t'
		return_str += 'P output: ' + str(self.P_output) + '\n\t\tmaximum: ' + str(self.P_max) + '\n\t\tminimum: ' + str(self.P_min) + '\n\t'
		return_str += 'Q output: ' + str(self.Q_output) + '\n\t\tmaximum: ' + str(self.Q_max) + '\n\t\tminimum: ' + str(self.Q_min) + '\n\t'
		# TODO add other characteristics
		return return_str

class Line(Asset):
	"""Define a distribution or transmission assets"""
	def __init__(self,  name = "Line",
						P_output = 0, # flow is from lower bus, i.e. negative if from higher to lower
						Q_output = 0, 
						P_nameplate_pos_neg = [1000, -1000],
						Q_nameplate_pos_neg = [1000, -1000],
						latency          = 0,
						P_time_ramp_up   = 0,
						P_time_ramp_down = 0,
						Q_time_ramp_up   = 0,
						Q_time_ramp_down = 0,
						P_real_time_max  = None,
						uncertainty      = None,
						inertia          = None,
						max_Q_support_pos_neg = None):

		self.max_Q_support_pos_neg = max_Q_support_pos_neg

		super().__init__(name, P_output, Q_output, P_nameplate_pos_neg, Q_nameplate_pos_neg, latency, P_time_ramp_up, P_time_ramp_down, Q_time_ramp_up, Q_time_ramp_down)
		
		
	# Line adaptive capacity function
	def calc_adaptive_capacity(self):
		"""Adaptive capacity of a line"""

		# The 3D domain bound based on the flexibility
		P = np.repeat(self.flex.get('P').reshape(-1,1), self.time_seconds.size, axis=1).T
		Q = np.repeat(self.flex.get('Q').reshape(-1,1), self.time_seconds.size, axis=1).T
		if self.max_Q_support_pos_neg is not None:
			Q = np.where(Q > self.max_Q_support_pos_neg[0], self.max_Q_support_pos_neg[0], Q)
			Q = np.where(Q < self.max_Q_support_pos_neg[1], self.max_Q_support_pos_neg[1], Q)

		self.ac = {'Uncertainty':False, 'P':P, 'Q':Q} 



class Dam(Asset):
	"""Define a dam generating asset"""
	
	def __init__(self,  name = 'Dam Asset', 
						P_output = 0, 
						Q_output = 0, 
						P_nameplate_pos_neg = [2000, 0], 
						Q_nameplate_pos_neg = [2000, -2000], 
						latency          = 120, 
						P_time_ramp_up   = 180, 
						P_time_ramp_down = 120, 
						Q_time_ramp_up   = 90, 
						Q_time_ramp_down = 90,
						P_real_time_max  = 100,
						uncertainty      = None, 
						inertia          = 100):

		super().__init__(name, P_output, Q_output, P_nameplate_pos_neg, Q_nameplate_pos_neg, latency, P_time_ramp_up, 
			                   P_time_ramp_down, Q_time_ramp_up, Q_time_ramp_down, P_real_time_max, uncertainty, inertia)
			
class Solar(Asset):
	"""Define a solar PV asset"""

	def __init__(self,  name = 'Solar Asset', 
						P_output = 0, 
						Q_output = 0, 
						P_nameplate_pos_neg = [100, 0], 
						Q_nameplate_pos_neg = [100, -100], 
						latency = 15, 
						P_time_ramp_up = 2, 
						P_time_ramp_down = 2, 
						Q_time_ramp_up = 2, 
						Q_time_ramp_down = 2,
						P_real_time_max=None,
						uncertainty=None):

		super().__init__(name, P_output, Q_output, P_nameplate_pos_neg, Q_nameplate_pos_neg, latency, 
							   P_time_ramp_up, P_time_ramp_down, Q_time_ramp_up, Q_time_ramp_down, P_real_time_max, uncertainty)

class Battery(Asset):
	"""Define a battery storage asset"""

	def __init__(self,  name = 'Battery Asset', 
						P_output = 0, 
						Q_output = 0, 
						P_nameplate_pos_neg = [100, -50], 
						Q_nameplate_pos_neg = [100, -100], 
						latency = 1, 
						P_time_ramp_up = 2, 
						P_time_ramp_down = 2, 
						Q_time_ramp_up = 2, 
						Q_time_ramp_down = 2,
						P_real_time_max = None,
						uncertainty = None,
						time_empty_2_full_charge_hrs = 2,
						time_full_charge_2_empty_hrs = 2,
						charge_percent = 50):

		self.time_empty_2_full_charge_hrs = time_empty_2_full_charge_hrs
		self.time_full_charge_2_empty_hrs = time_full_charge_2_empty_hrs
		self.charge_percent = charge_percent

		super().__init__(name, P_output, Q_output, P_nameplate_pos_neg, Q_nameplate_pos_neg, latency, 
							   P_time_ramp_up, P_time_ramp_down, Q_time_ramp_up, Q_time_ramp_down, P_real_time_max, uncertainty)

		
	def calc_energy_limits(self):

		if self.charge_percent is not None:

			time_until_empty_sec = self.time_full_charge_2_empty_hrs * self.charge_percent/100 * 3600
			time_until_full_sec =  self.time_empty_2_full_charge_hrs * (1 - self.charge_percent/100) * 3600
			
			P_total = np.cumsum(self.ac.get('P') + self.P_output, axis=0) * self.time_seconds / 3600
			
			# Find index where energy limit is reached
			if time_until_empty_sec < self.time_seconds[-1]:
				idx_up   = np.argmax(self.time_seconds >= time_until_empty_sec)

				#if idx_up < P_total[:,0].size:
				P_at_time = P_total[idx_up, 0]
				self.ac.update({'P': np.where(P_total > P_at_time, -self.P_output, self.ac.get('P'))})
				self.ac.update({'Q': np.where(P_total > P_at_time, -self.Q_output, self.ac.get('Q'))})


			if time_until_full_sec < self.time_seconds[-1]:
				idx_down = np.argmax(self.time_seconds >= time_until_full_sec) 
				
				#if idx_down < P_total[:,0].size:
				P_at_time = P_total[idx_down, 2*self.number_radial_pts]
				self.ac.update({'P':np.where(P_total < P_at_time, -self.P_output, self.ac.get('P'))})
				self.ac.update({'Q':np.where(P_total < P_at_time, -self.Q_output, self.ac.get('Q'))})

class Dam_pump_storage(Asset):
	"""Define a dam pump storage asset"""

	def __init__(self,  name = 'Hydro Pump Storage', 
						P_output = 0, 
						Q_output = 0, 
						P_nameplate_pos_neg = [0, -100], 
						Q_nameplate_pos_neg = [100, -100], 
						latency = 120, 
						P_time_ramp_up = 120, 
						P_time_ramp_down = 120, 
						Q_time_ramp_up = 120, 
						Q_time_ramp_down = 120,
						P_real_time_max = None,
						uncertainty = None,
						time_until_full_min = None):

		self.time_until_full_min = time_until_full_min

		super().__init__(name, -P_output, Q_output, P_nameplate_pos_neg, Q_nameplate_pos_neg, latency, 
							    P_time_ramp_up, P_time_ramp_down, Q_time_ramp_up, Q_time_ramp_down, P_real_time_max, uncertainty)

	def calc_energy_limits(self):
		if self.time_until_full_min is not None:

			P_total = np.cumsum(self.P_adaptive_capacity_3D + self.P_output, axis=0) * self.time_seconds / 3600
			idx = np.argmax(time_seconds >= (time_until_full_min * 60))

			if idx < P_total[:,0].size:
				P_at_time = P_total[idx, 2*self.number_radial_pts]
				self.P_adaptive_capacity_3D = np.where(P_total < P_at_time, -self.P_output, self.P_adaptive_capacity_3D)
				self.Q_adaptive_capacity_3D = np.where(P_total < P_at_time, -self.Q_output, self.Q_adaptive_capacity_3D)

class Wind(Asset):
	"""Define a wind turbine asset"""

	def __init__(self,  name = 'Wind Asset', 
						P_output = 0.8, 
						Q_output = 0, 
						P_nameplate_pos_neg = [1, 0], 
						Q_nameplate_pos_neg = [1, -1], 
						latency = 2, 
						P_time_ramp_up = 180, 
						P_time_ramp_down = 120, 
						Q_time_ramp_up = 30, 
						Q_time_ramp_down = 30,
						P_real_time_max = None,
						uncertainty = None):
		
		super().__init__(name, P_output, Q_output, P_nameplate_pos_neg, Q_nameplate_pos_neg, latency, 
							   P_time_ramp_up, P_time_ramp_down, Q_time_ramp_up, Q_time_ramp_down, P_real_time_max, uncertainty)

class Turbine(Asset):
	"""Define a gas turbine generation unit"""

	def __init__(self, name = 'Gas Turbine', 
					  P_output = 0, 
					  Q_output = 0, 
					  P_nameplate_pos_neg = [20, 0], 
					  Q_nameplate_pos_neg = [20, -20], 
					  latency = 300, 
					  P_time_ramp_up = 1200, 
					  P_time_ramp_down = 600, 
					  Q_time_ramp_up = 300, 
					  Q_time_ramp_down = 300,
					  P_real_time_max = None,
					  uncertainty = None,
					  inertia = 100):
		super().__init__(name, P_output, Q_output, P_nameplate_pos_neg, Q_nameplate_pos_neg, latency, 
							   P_time_ramp_up, P_time_ramp_down, Q_time_ramp_up, Q_time_ramp_down, P_real_time_max, uncertainty, inertia)


