from numpy import c_
from system_params import * 

def optimize_computation_task(idx_user, f_max, psi, queue_t, d_t):
	opt_tasks = np.zeros((no_users)) 
	opt_energy = np.zeros((no_users))
	no_opt_users = len(idx_user)
	obj_value = 0 
	# queue_t: numpy array 
	# idx_user: numpy array np.where(off == 1)[0]
	queue_opt_users = queue_t[idx_user]
	dt_opt_users = d_t[idx_user]

	if no_opt_users != 0: 
		bounded_freq = np.minimum(f_max, queue_opt_users*F/ts_duration)
		freq = np.sqrt((queue_opt_users + dt_opt_users)/(3*LYA_V*KAPPA*PSI*F))
		freq_opt = np.minimum(freq, bounded_freq)

		opt_tasks[idx_user] = np.round(freq_opt*ts_duration/F)
		opt_energy = KAPPA * ((opt_tasks*F/ts_duration)**3) * ts_duration
		
		obj_value = np.sum(LYA_V*psi*opt_energy - queue_t*opt_tasks)
	
	return obj_value, opt_tasks, opt_energy

def opt_commun_tasks_eqbw(idx_user, Q_t, L_t, gain_t): 
	opt_tasks = np.zeros((no_users)) 
	opt_energy = np.zeros((no_users))
	no_opt_users = len(idx_user)
	obj_value = 0 

	# queue_t: numpy array 
	# idx_user: numpy array np.where(off == 1)[0]

	idx_queue = Q_t[idx_user] > L_t[idx_user]
	idx_opt = idx_user[idx_queue]

	q_opt_users = Q_t[idx_opt]
	l_opt_users = L_t[idx_opt]
	gain_opt_users = gain_t[idx_opt]
	
	if no_opt_users != 0: 
		eqbw = BW_W/no_opt_users 
		bounded_b = np.minimum(q_opt_users, np.round(eqbw*ts_duration/R*np.log2(1 + pi_0*gain_opt_users/N0/eqbw)))
		opt_tasks[idx_opt] = np.maximum(0, \
			np.minimum(bounded_b, \
			np.round(eqbw*ts_duration/R*np.log2((q_opt_users - l_opt_users)*gain_opt_users/(LYA_V*N0*R*np.log(2))))))	
		
		opt_energy[idx_opt] = (N0*eqbw*ts_duration/gain_opt_users)*(2**(opt_tasks[idx_opt] *R/eqbw/ts_duration) - 1)

		obj_value = np.sum(- opt_tasks*(Q_t - L_t) + LYA_V*opt_energy)

	return obj_value, opt_tasks, opt_energy

def resource_allocation(off_decsion, Q, L, gain, D):
	local_ue = np.where(off_decsion == 0)[0]
	off_ue = np.where(off_decsion == 1)[0]
	obj1, a_t, energy_ue_pro = optimize_computation_task(local_ue, f_max=fi_0, psi=1, queue_t=Q, d_t=D)
	obj2, b_t, energy_ue_off = opt_commun_tasks_eqbw(off_ue, Q_t=Q, L_t=L, gain_t=gain)
	obj3, c_t, energy_uav_pro = optimize_computation_task(off_ue, f_max=fu_0, psi= PSI, queue_t=L, d_t=D)

	obj_value = obj1 + obj2 + obj3
	E_ue = energy_ue_pro + energy_ue_off 
	E_uav = energy_uav_pro

	return obj_value, a_t, b_t, c_t, E_ue, E_uav

def preprocessing(data_in):
    # create scaler 
    scaler = MinMaxScaler()
    data = np.reshape(data_in, (-1, 1))
    # fit scaler on data 
    scaler.fit(data)
    normalized = scaler.transform(data)
    normalized = normalized.reshape(1, -1)
    return normalized

def gen_actions_bf(no_users = 5):
  import itertools
  actions = np.array(list(itertools.product([0, 1], repeat=no_users))) # (32, 5)
  return actions