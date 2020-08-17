import numpy as np
from .nqs import *

class Sampler:
	def __init__(self,wf,N_up,N_down,writestates=True):
		self.wf = wf # nqs wave function
		self.N_orbitals = self.wf.N_orbitals
		self.N_up = N_up
		self.N_down = N_down
		self.spin_prob = [N_down/(N_up+N_down),1-N_down/(N_up+N_down)]
		self.excitation = None
		self.state = self.wf.state

		self.writestates = writestates
		self.nexcites = 0
		self.accepted = 0

	def init_state(self,mod='low'):
		if mod not in ['low','rand','high']:
			raise RuntimeError('''Initialization mod only support ['low','rand','high']''')

		N_orbitals = self.N_orbitals
		N_up = self.N_up
		N_down = self.N_down

		if mod=='high':
			config = np.ones(2*N_orbitals)
			config[:N_orbitals-N_up] = -1
			config[N_orbitals:-N_down] = -1

			index = {}
			index['vir_up'] = np.arange(N_orbitals-N_up)
			index['occ_up'] = np.arange(N_orbitals-N_up,N_orbitals)
			index['vir_down'] = np.arange(N_orbitals-N_down)
			index['occ_down'] = np.arange(N_orbitals-N_down,N_orbitals)

		elif mod=='low':
			config = np.ones(2*N_orbitals)
			config[N_up:N_orbitals] = -1
			config[N_orbitals+N_down:] = -1

			index = {}
			index['vir_up'] = np.arange(N_up,N_orbitals)
			index['occ_up'] = np.arange(N_up)
			index['vir_down'] = np.arange(N_down,N_orbitals)
			index['occ_down'] = np.arange(N_down)

		else:
			config = -1 * np.ones(2*N_orbitals)

			index = {}
			index['occ_up'] = np.random.choice(np.arange(N_orbitals),N_up,replace=False)
			index['occ_down'] = np.random.choice(np.arange(N_orbitals),N_down,replace=False)

			config[index['occ_up']] = 1
			config[N_orbitals + index['occ_down']] = 1

			index['vir_up'] = np.where(config[:N_orbitals] == -1)[0]
			index['vir_down'] = np.where(config[N_orbitals:] == -1)[0] # no need to minus N_orbitals

		self.state = (config,index)
		self.wf.init_table(self.state)

	def reset_count(self):
		self.nexcites = 0
		self.accepted = 0

	def acceptance(self):
		return self.accepted / self.nexcites

	def elec_excit(self):
		if self.state == None:
			self.init_state()
		N_orbitals = self.N_orbitals
		N_up = self.N_up
		N_down = self.N_down

		if np.random.choice([0,1],p=self.spin_prob):
			elec = 0 # electron spin up
			inx_occ = np.random.randint(0,N_up)
			occ = self.state[1]['occ_up'][inx_occ]
			inx_vir = np.random.randint(0,N_orbitals-N_up)
			vir = self.state[1]['vir_up'][inx_vir]
		else:
			elec = 1 # electron spin down
			inx_occ = np.random.randint(0,N_down)
			occ = self.state[1]['occ_down'][inx_occ]
			inx_vir = np.random.randint(0,N_orbitals-N_down)
			vir = self.state[1]['vir_down'][inx_vir]

		self.excitation = [elec,occ,vir,inx_occ,inx_vir]

	def move(self,filename=None):
		self.elec_excit()
		accept_prob = np.abs(self.wf.prob(self.excitation)) ** 2
		accept_prob = min(1,accept_prob)
		if accept_prob > np.random.random():
			self.wf.update_nqs(self.excitation)
			self.state = self.wf.state
			self.accepted += 1
		self.nexcites +=1

		if (self.writestates) and (filename!=None):
			with open(filename,'a',newline=None) as fout:
				fout.write(' '.join(map(str,self.state[0]))+'\n')

	def thermalize(self,nmoves=None,filename=None):
		if nmoves==None:
			nmoves= (self.N_up+self.N_down) * 5
		for sweep in range(nmoves):
			self.move(filename)
