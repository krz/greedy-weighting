import numpy as np
import scipy.stats as sc
import time



def metric_spearman(x,Y):
	"""some docstring is missing"""
	return sc.stats.spearmanr(x, Y)[0]

def conv_manhattan(n, vec, eps):
	"""some docstring is missing"""
	return n >= 1/eps

def conv_euclid(n, vec, eps):
	"""some docstring is missing"""
	return np.linalg.norm(vec) >= 1/eps

def greedy_opt(X, Y, metric=metric_spearman, converged=conv_manhattan, eps=1e-2):
	"""some docstring is missing"""

	weights = np.zeros(X.shape[1], dtype=int) # weights for the columns of X
	sums_transposed = np.zeros(X.transpose().shape)

	num_weights = 0 # integer makes increment fast and stable
	best = 0
	t0 = time.time()
	#old_m = -1 # for metric convergence
	#new_m = -1 # for metric convergence
	#while not converged(num_weights, weights, 0.01) or abs((new_m - old_m) / new_m) > 0.001:#metric convergence
	while not converged(num_weights, weights, eps):
		num_weights += 1
		sums_transposed = sums_transposed[best, :] + X.transpose()
		err = [metric(sums_transposed[i,:] / float(num_weights), Y) 
			for i in xrange(sums_transposed.shape[0])]
		best = np.argmax(err)
		weights[best] += 1
		#old_m = new_m # for metric convergence
		#new_m = err[best] # for metric convergence
	t = time.time()
	print('took {} seconds.'.format(t-t0))
	print(num_weights)
	print(weights)
	#print((new_m - old_m) / new_m)
	########################################################################
	#max_weighted = err[best]
	#err_pure = [metric(X[:,i], Y) for i in range(X.shape[1])]
	#max_pure = np.amax(err_pure)
	#if (max_pure > max_weighted):
		#print("optimization did not lead to improved result")
	#print('best orig/weighted are: {}/{}'.format(max_pure, max_weighted))
	########################################################################
	return weights/float(num_weights)

