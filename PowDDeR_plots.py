import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from helper_functions import *

def plot_domain(self):
	"""2D plot of the domain limit"""

	if hasattr(self, 'domain'):
		plt.plot(self.domain.get('P'), self.domain.get('Q'), '.', label="Domain Boundary")   
		plt.title(self.name + ' Domain Limits')
		plt.xlabel('Real Power')
		plt.ylabel('Reactive Power')
		plt.show()
	else:
		try:
			print(self.name + ' does not have domain attribute')
		except:
			print('Asset does not have a domain attribute')
		print('\tNote: bus, ASR, and dist. system does not track total domain')

 
def plot_flex(self, time_minutes=5):
	"""2D plot of the asset flexibility"""

	if isinstance(time_minutes, int):
		P, Q = get_flexibility_at_time(self, time_minutes)
		plt.plot(P, Q)
	else:
		leg = []
		for i, time in enumerate(time_minutes):
			P, Q = get_flexibility_at_time(self, time_minutes[i])
			plt.plot(P, Q)
			if time_minutes[i] >= 1:
				leg.append(str(time_minutes[i]) + 'min')
			else:
				leg.append(str(int(time_minutes[i]*60)) + 'sec')
		
		plt.legend(leg, loc='upper left')

	plt.title(self.name + ' Adaptive Capacity')
	plt.xlabel('Real Power (MW)')
	plt.ylabel('Reactive Power (MVAR)')

	plt.show()


def plot_ramp_rates(self):
	"""2D plot of the ramp rates wrt flexibility"""

	# Only assets have the ramp rate in real and reactive power
	if hasattr(self, 'Pt_up'):
		plt.plot(self.time_minutes[:,0], self.ramp.get('P_up'),   label='Real Up')
		plt.plot(self.time_minutes[:,0], self.ramp.get('P_down'), label='Real Down')
		plt.plot(self.time_minutes[:,0], self.ramp.get('Q_up'),   label='Reactive Up')
		plt.plot(self.time_minutes[:,0], self.ramp.get('Q_down'), label='Reactive Down')
	        
	# Bus, ASR, and system need to pull values from the adaptive capacity
	else:
		n = int(len(self.pfa_radians)/4)
		ac = get_aggregation_ac(self)
		plt.plot(self.time_minutes[:,0], np.maximum(ac.get('P')[:,0],   ac.get('P')[:,-1]),    label='Real Up')
		plt.plot(self.time_minutes[:,0], np.maximum(ac.get('P')[:,2*n], ac.get('P')[:,2*n+1]), label='Real Down')
		plt.plot(self.time_minutes[:,0], np.maximum(ac.get('Q')[:,n],   ac.get('Q')[:,n+1]),   label='Reactive Up')
		plt.plot(self.time_minutes[:,0], np.maximum(ac.get('Q')[:,3*n], ac.get('Q')[:,3*n+1]), label='Reactive Down')
	        
	plt.title(self.name + ' Ramp Rates From Operating Point')
	plt.xlabel('Time (min)')
	plt.ylabel('Power')
	
	plt.legend()
	plt.show()


def plot_ac(self, color='blue', ac='expected'):

	plt.figure(figsize=(6,7))
	#plt.suptitle(self.name + ' Adaptive Capacity', fontsize=16)

	adaptive_capacity = get_aggregation_ac(self)


	if ac == 'expected':
		x = adaptive_capacity.get('P')
	elif ac == 'pos':
		x = adaptive_capacity.get('P_pos_unc')
	elif ac == 'neg':
		x = adaptive_capacity.get('P_neg_unc')
	else:
		print('Unable to get adaptive capacity.')
		return

	y = adaptive_capacity.get('Q')
	# Append the first column to the end to close the surface	
	try:
		x = np.append(x, x[:,0].reshape(-1,1), axis=1)
	except:
		print(f'{self.name} does not have any adaptive capacity')
		return

	y = np.append(y, y[:,0].reshape(-1,1), axis=1)
	z = np.append(self.time_minutes, self.time_minutes[:,0].reshape(-1,1), axis=1)

	ax = plt.axes(projection='3d')

	if hasattr(self, 'max_Q_support_pos_neg'):
		ax.set_title(f'{self.name} Capacity', fontsize=15)
	else:
		ax.set_title(f'{self.name} Adaptive Capacity', fontsize=15)
	ax.set_xlabel('$P_{AC} (MW)$')
	ax.set_ylabel('$Q_{AC} (MVAR)$')
	ax.set_zlabel('Time (min)')

	if z[-1,0] < 10:
		z = z*60
		ax.set_zlabel('Time (s)')

	ax.plot_surface(x, y, z, edgecolor='none', color=color, alpha=0.6, shade=True, rcount=200, ccount=200)
	plt.tight_layout()
	plt.show()


def plot_two_ac(one, two):

	plt.figure(figsize=(6,7))
	ax = plt.axes(projection='3d')

	adaptive_capacity = get_aggregation_ac(one)

	x = adaptive_capacity.get('P')
	y = adaptive_capacity.get('Q')

	# Append the first column to the end to close the surface	
	x1 = np.append(x, x[:,0].reshape(-1,1), axis=1)
	y1 = np.append(y, y[:,0].reshape(-1,1), axis=1)
	z = np.append(one.time_minutes, one.time_minutes[:,0].reshape(-1,1), axis=1)

	ax.plot_surface(x1, y1, z, edgecolor='none', alpha=0.3)


	adaptive_capacity = get_aggregation_ac(two)

	x = adaptive_capacity.get('P')
	y = adaptive_capacity.get('Q')

	# Append the first column to the end to close the surface	
	x2 = np.append(x, x[:,0].reshape(-1,1), axis=1)
	y2 = np.append(y, y[:,0].reshape(-1,1), axis=1)
	z2 = np.append(two.time_minutes, two.time_minutes[:,0].reshape(-1,1), axis=1)

	ax.plot_surface(x2, y2, z2, edgecolor='none', alpha=0.3)

	x_min = np.empty_like(z)
	y_min = np.empty_like(z)

	x_min[:, one.i.get('q1')] =  np.minimum(       x1[:, one.i.get('q1')],         x2[:, one.i.get('q1')])
	x_min[:, one.i.get('q2')] = -np.minimum(np.abs(x1[:, one.i.get('q2')]), np.abs(x2[:, one.i.get('q2')]))
	x_min[:, one.i.get('q3')] = -np.minimum(np.abs(x1[:, one.i.get('q3')]), np.abs(x2[:, one.i.get('q3')]))
	x_min[:, one.i.get('q4')] =  np.minimum(       x1[:, one.i.get('q4')],         x2[:, one.i.get('q4')])

	y_min[:, one.i.get('q1')] =  np.minimum(       y1[:, one.i.get('q1')],         y2[:, one.i.get('q1')])
	y_min[:, one.i.get('q2')] =  np.minimum(       y1[:, one.i.get('q2')],         y2[:, one.i.get('q2')])
	y_min[:, one.i.get('q3')] = -np.minimum(np.abs(y1[:, one.i.get('q3')]), np.abs(y2[:, one.i.get('q3')]))
	y_min[:, one.i.get('q4')] = -np.minimum(np.abs(y1[:, one.i.get('q4')]), np.abs(y2[:, one.i.get('q4')]))



	ax.plot_surface(x_min, y_min, z, edgecolor='none', alpha=0.95)

	ax.set_title('Adaptive Capacity', fontsize=14)
	ax.set_xlabel('$P_{AC}$')
	ax.set_ylabel('$Q_{AC}$')
	ax.set_zlabel('Time (min)')
	
	plt.show()


def get_aggregation_ac(self):

	if hasattr(self, 'ac'):
		return {'P':self.ac.get('P'), 'Q':self.ac.get('Q')}
	else:
		P = Q = P_pos = P_neg = 0
		uncertainty = False

		for i in range(0, self.number_of_assets):
			if self.list_of_assets[i].uncertainty is None:
				P += self.list_of_assets[i].ac.get('P')
				Q += self.list_of_assets[i].ac.get('Q')
				P_pos += self.list_of_assets[i].ac.get('P')
				P_neg += self.list_of_assets[i].ac.get('P')
			else:
				P += self.list_of_assets[i].ac.get('P')
				Q += self.list_of_assets[i].ac.get('Q')
				P_pos += self.list_of_assets[i].ac.get('P_pos_unc')
				P_neg += self.list_of_assets[i].ac.get('P_neg_unc')
				uncertainty = True

		adaptive_capacity = {'Uncertainty':uncertainty, 'P':P, 'Q':Q, 'P_pos_unc':P_pos, 'P_neg_unc':P_neg}

	return adaptive_capacity