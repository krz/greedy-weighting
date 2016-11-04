import numpy

class Sampler(object):

	def __init__(self, size, steps_per_dim):
		self.steps_per_dim = steps_per_dim
		self.sample = numpy.zeros(size, dtype=int)
		self.size = size
		self.next = self.__initial_sample 

	def __initial_sample(self):
		self.sample[-1] = self.steps_per_dim
		self.next = self.__next
		return True

	def __correct_sample_dim(self, dim):
		changed = True
		if self.sample[dim] < self.steps_per_dim - self.sample[:dim].sum():
			self.sample[dim] += 1
		elif dim != 0:
			self.sample[dim] = 0
			changed = self.__correct_sample_dim(dim-1)
		else: # dim = 0 and entry==steps_per_dim!
			changed = False
		return changed

	def __next(self):
		changed = self.__correct_sample_dim(self.size-2)
		self.sample[-1] = self.steps_per_dim - self.sample[:-1].sum()
		if changed:
			self.__next = lambda : False
		return changed


	def next_sequence(self, num, samples):
		while self.next() and num > 0:
			samples.append(self.sample.copy())
			num -= 1
		return num == 0


def simple_test_sequence():
	steps_per_dim = 100
	sampler = Sampler(size=3, steps_per_dim=steps_per_dim)
	some_left = True
	while some_left:
		seq = []
		some_left = sampler.next_sequence(10, seq)
		print()
		print(seq)

def simple_test():
	steps_per_dim = 100
	sampler = Sampler(size=3, steps_per_dim=steps_per_dim)
	print(sampler.sample)
	s = 0
	while sampler.next():
		pass
		#print(sampler.sample)


	steps_per_dim = 100
	sampler = Sampler(size=4, steps_per_dim=steps_per_dim)
	print(sampler.sample)
	s = 0
	while sampler.next():
		print(sampler.sample)
		pass



if __name__=='__main__':
	import time
	t0 = time.time()
	simple_test()
	print('took {} seconds.'.format(time.time()-t0))
	#t0 = time.time()
	#simple_test_sequence()
	#print('took {} seconds.'.format(time.time()-t0))
