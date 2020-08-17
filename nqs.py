import numpy as np

class Nqs:
	def __init__(self,N_orbitals,alpha):
		# alpha is N_hidden/N_input
		self.N_orbitals = N_orbitals
		self.In = N_orbitals * 2
		self.Hi = int(self.In * alpha)
		self.W = np.zeros((self.In,self.Hi),dtype=np.complex128)
		self.a = np.zeros(self.In,dtype=np.complex128)
		self.b = np.zeros(self.Hi,dtype=np.complex128)

		self.state = None
		# theta = b[i] + \sum_j (a[j]*W[j,i])
		self.theta = np.zeros(self.Hi,dtype=np.complex128)
		# logpsi_2 = log(2*cosh(theta))
		self.logpsi_2 = np.zeros(self.Hi,dtype=np.complex128)
		

	def init_table(self,state):
		self.state = state
		self.theta = self.b + np.dot(state[0],self.W)
		self.logpsi_2 = np.log(2*np.cosh(self.theta))


	def prob(self,excitation):
		# one electron excitation
		occ = excitation[0] * self.N_orbitals + excitation[1]
		vir = excitation[0] * self.N_orbitals + excitation[2]

		logprob = 2*self.a[vir] - 2*self.a[occ]
		logprob += np.sum(np.log(2*np.cosh( (self.theta+2*self.W[vir]-2*self.W[occ]) ))\
			- self.logpsi_2)

		return np.exp(logprob)

	def update_nqs(self,excitation):
		# one electron excitation
		occ = excitation[0] * self.N_orbitals + excitation[1]
		vir = excitation[0] * self.N_orbitals + excitation[2]

		# update look-up table
		self.theta = self.theta + 2*self.W[vir] - 2*self.W[occ]
		self.logpsi_2 = np.log(2*np.cosh(self.theta))

		# update state
		self.state[0][occ] *= -1
		self.state[0][vir] *= -1

		if excitation[0]:
			# excitation[0]==1, excite spin down electron
			self.state[1]['occ_down'][excitation[3]] = excitation[2]
			self.state[1]['vir_down'][excitation[4]] = excitation[1]
		else:
			# excitation[0]==0, excite spin up electron
			self.state[1]['occ_up'][excitation[3]] = excitation[2]
			self.state[1]['vir_up'][excitation[4]] = excitation[1]

	def load_parameters(self,filename):
		tmp = np.load(filename)
		self.a = tmp['a']
		self.b = tmp['b']
		self.W = tmp['W']
		self.In = len(self.a)
		self.Hi = len(self.b)
		self.N_orbitals = self.In // 2

	def save_parameters(self,filename):
		np.savez(filename, a=self.a, b=self.b, W=self.W)

