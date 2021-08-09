import numpy as np
import os


def get_flexibility_at_time(self, time_minutes=1):
	idx_flex = np.argmax(self.time_minutes[:,0] > time_minutes)
	if hasattr(self, 'ac'):
		P_flex = self.ac.get('P')[idx_flex, :]
		Q_flex = self.ac.get('Q')[idx_flex, :]
		return np.append(P_flex, P_flex[0]), np.append(Q_flex, Q_flex[0]) 
	else:
		ac = get_aggregation_ac(self)
		P_flex = ac.get('P')[idx_flex, :]
		Q_flex = ac.get('Q')[idx_flex, :]
		return np.append(P_flex, P_flex[0]), np.append(Q_flex, Q_flex[0]) 


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


def get_connected_ac(self):
	# If it is an asset
	if hasattr(self, 'P_adaptive_capacity_3D'):
		print('Asset has no connected adaptive capacity')
		return 
	
	con_ac = {'P':0, 'Q':0}

	for bus in self.connected_to_buses:
		print(bus[0].name)
		asset_ac = get_aggregation_ac(bus[0])
		line_ac = bus[1].ac


def save_plot(self):

	fig_path = os.path.join('res','fig','{}'.format(date.today()))
	if not os.path.exists(fig_path):
		os.makedirs(fig_path)

	plt.savefig(os.path.join(fig_path,'{}.pdf'.format(self.name)), bbox_inches='tight')



def save_csv(self, fileName):
	"""Save the adaptive capacity of an asset or ASR in .csv file"""

	csv_path = os.path.join('res','data','Surface_Results','{}'.format(date.today()))
	if not os.path.exists(csv_path):
		os.makedirs(csv_path)

	np.savetxt(os.path.join(csv_path,'{}_cyl_dist.csv'.format(fileName)), self.S_adaptive_capacity_3D, delimiter=",")
	np.savetxt(os.path.join(csv_path,'{}_P_adpt_cap.csv'.format(fileName)), np.cos(self.pfa_radians) * self.S_adaptive_capacity_3D, delimiter=",")
	np.savetxt(os.path.join(csv_path,'{}_Q_adpt_cap.csv'.format(fileName)), np.sin(self.pfa_radians) * self.S_adaptive_capacity_3D, delimiter=",")
	np.savetxt(os.path.join(csv_path,'{}_time.csv'.format(fileName)), self.time_minutes, delimiter=",")


def save_json(self, fileName):

	data ={'adaptive_capacity':{'x': self.ac.get('P').tolist(),
	                            'y': self.ac.get('Q').tolist(),
	                            'z': self.time_minutes.tolist()}}

	with open(fileName, 'w') as outfile:
		json.dump(data, outfile)


