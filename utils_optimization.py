from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cmath import sqrt
from numpy import arange, c_
from system_params import * 
from scipy.optimize import minimize, Bounds, LinearConstraint

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

		opt_tasks[idx_user] = np.floor(freq_opt*ts_duration/F)
		opt_energy = KAPPA * ((opt_tasks*F/ts_duration)**3) * ts_duration
		
		obj_value = np.sum(LYA_V*psi*opt_energy - (queue_t + d_t)*opt_tasks)
	
	return obj_value, opt_tasks, opt_energy

def optimize_uav_freq(idx_user, f_max, psi, queue_t, d_t):
	# no_opt_user = len(idx_user)
	# f_0ue = f_max/no_opt_user
	# return optimize_computation_task(idx_user, f_0ue, psi, queue_t, d_t)

	opt_value = np.sqrt(queue_t + d_t)
	if np.sum(opt_value) != 0: 
		opt_scale = opt_value/np.sum(opt_value)
	else: 
		opt_scale = 1/no_users 

	obj_value, opt_tasks, opt_energy = optimize_computation_task(idx_user, f_max*opt_scale, psi, queue_t, d_t)
		# opt_energy = KAPPA * np.sum((opt_tasks*F/ts_duration))**3 * ts_duration
		# obj_value = LYA_V*psi*opt_energy - np.sum((queue_t + d_t)*opt_tasks)

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
		bounded_b = np.minimum(q_opt_users, np.floor(eqbw*ts_duration/R*np.log2(1 + pi_0*gain_opt_users/N0/eqbw)))
		opt_tasks[idx_opt] = np.maximum(0, \
			np.minimum(bounded_b, \
			np.floor(eqbw*ts_duration/R*np.log2((q_opt_users - l_opt_users)*gain_opt_users/(LYA_V*N0*R*np.log(2))))))	
		
		opt_energy[idx_opt] = (N0*eqbw*ts_duration/gain_opt_users)*(2**(opt_tasks[idx_opt] *R/eqbw/ts_duration) - 1)

		obj_value = np.sum(- opt_tasks*(Q_t - L_t) + LYA_V*opt_energy)

	return obj_value, opt_tasks, opt_energy

def optimize_computation_uav(f_max, psi, l_t, d_t):
	def objective_func(f_iU):
		return np.sum(-(l_t + d_t)*f_iU*ts_duration/F + LYA_V*psi*KAPPA*ts_duration*(f_iU**3))

	def obj_der(f_iU):
		return -(l_t + d_t)*ts_duration/F + LYA_V*psi*KAPPA*3*(f_iU**2)

	def obj_hess(f_iU):
		return LYA_V*KAPPA*psi*ts_duration*6*f_iU

	x = np.ones((no_users))
	bounds = Bounds(x*0, x*l_t*F/ts_duration)
	linear_constraint = LinearConstraint(x, 0, f_max)

	f_iU_0 = np.ones(no_users)
	res = minimize(objective_func, f_iU_0, method='trust-constr', jac=obj_der, hess=obj_hess, tol=1e-7,
	            constraints=[linear_constraint], bounds=bounds)

	# update uav frequency 
	f_u = res.x
	obj_value = res.fun 
	opt_tasks = np.floor(f_u*ts_duration/F)
	opt_energy = KAPPA*((opt_tasks*F/ts_duration)**3)*ts_duration
	return obj_value, opt_tasks, opt_energy

def resource_allocation(off_decsion, Q, L, gain, D):
	local_ue = np.where(off_decsion == 0)[0]
	off_ue = np.where(off_decsion == 1)[0]
	obj1, a_t, energy_ue_pro = optimize_computation_task(local_ue, f_max=fi_0, psi=1, queue_t=Q, d_t=D)
	obj2, b_t, energy_ue_off = opt_commun_tasks_eqbw(off_ue, Q_t=Q, L_t=L, gain_t=gain)
	# obj3, c_t, energy_uav_pro = optimize_computation_task(np.arange(no_users), f_max=f_iU_0, psi= PSI, queue_t=L, d_t=D)
	# obj3, c_t, energy_uav_pro = optimize_computation_uav(f_max=f_iU_0, psi= PSI, l_t=L, d_t=D)
	obj3, c_t, energy_uav_pro = optimize_uav_freq(np.arange(no_users), f_max=f_iU_0, psi= PSI, queue_t=L, d_t=D)


	obj_value = obj1 + obj2 + obj3
	E_ue = energy_ue_pro + energy_ue_off 
	E_uav = energy_uav_pro

	return obj_value, a_t, b_t, c_t, energy_ue_pro, energy_ue_off, energy_uav_pro

def preprocessing(data_in):
    scaler = StandardScaler()
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