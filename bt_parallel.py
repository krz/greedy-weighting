import multiprocessing
import scipy.stats as sc
import numpy as np
import bt_basis


data = None
vector = None

def eval_spearman(samples, max_val):
	"""some docstring is missing"""
	max_val = 0
	max_entry = None
	for entry in samples:

		mix = np.dot(data, entry)
		val = sc.stats.spearmanr(mix, vector)[0]

		if val > max_val:
			max_val = val
			max_entry = entry
	return max_val, max_entry



if __name__=='__main__':


	data = np.loadtxt('gimd.csv', delimiter=',')
	vector = np.loadtxt('smr.csv', delimiter=',')

	#steps_per_dim = 20 # discretization for each component
	steps_per_dim = 100 # discretization for each component
	sampler = bt_basis.Sampler(7, steps_per_dim)
	seq_len = 1e+4 # length of sequence for each core
	cores = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=cores)

	max_val = 0
	max_entry = None
	some_left = True
	while some_left:
		#print('.')
		results = []
		# repetitions -> length of tmp array to be evalutated
		for _ in range(cores):
			print(_)
			#print(_)
			seq = []
			some_left = sampler.next_sequence(seq_len, seq)
			#print(seq)
			samples = np.array(seq) / float (steps_per_dim)
			results.append(pool.apply_async(eval_spearman, args=(samples, max_val)))
			if not some_left:
				break

		for r in results:
			#print(r.get())
			val, entry = r.get()
			if val > max_val:
				max_val = val
				max_entry = entry
	
	print(max_val, max_entry)

