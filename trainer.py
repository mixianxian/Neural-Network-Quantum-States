import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from joblib import Parallel, delayed

from .sampler import *

class Trainer:
	def __init__(self, mol_info, reg_list=(100,0.9,1e-4), cores=8):
		self.h1e, self.eri, self.N_up, self.N_down = mol_info
		self.reg_list = reg_list
		self.step_count = 0
		self.nvar = 0
		self.parallel_cores = cores

	def train(self,wf,ecore,batch_size=100,num_steps=1000,init_state=None,\
		print_freq=20,state_file=None,param_file=None,out_freq=20):
		state = init_state
		# Energy list
		elist = np.zeros(num_steps+1,dtype=np.complex128)
		# early convergence
		convergence = 0
		for step in range(num_steps+1):
			updates, state, elist[step] = self.update_vector(wf,state,batch_size,gamma_fun(step),step,state_file)
			elist[step] += ecore
			self.apply_update(updates,wf)

			if (step!=0) and (abs(elist[step]-elist[step-1]) < 1e-3):
				convergence += 1
				if convergence > 10:
					print('Completed training step {}'.format(step))
					print("Current energy: {}".format(elist[step]))
					print("Energy converged! Program ends early!")
					break
			else:
				convergence = 0

			if (step % print_freq == 0) and (print_freq > 0):
				print('Completed training step {}'.format(step))
				print("Current energy: {}".format(elist[step]))

			if (out_freq>0) and (step%out_freq==0):
				wf.save_parameters(param_file+'Batch{}_Step{}'.format(batch_size,step))

		return wf, elist

	def eval(self,wf,ecore,batch_size=20000,init_state=None,param_file=None):
		if param_file != None:
			wf.load_parameters(param_file)
		if init_state == None:
			samp = Sampler(wf,self.N_up,self.N_down)
			samp.init_state('low')
		else:
			wf.init_table(init_state)
			samp = Sampler(wf,self.N_up,self.N_down)

		results = Parallel(n_jobs=self.parallel_cores)(\
			delayed(get_E)(samp,self) for batch in range(batch_size))

		return np.mean(results)+ecore


	def apply_update(self,updates,wf):
		wf.a += updates[0:wf.In]
		wf.b += updates[wf.In:wf.In+wf.Hi]
		wf.W += np.reshape(updates[wf.In+wf.Hi:],wf.W.shape)

	def update_vector(self,wf,init_state,batch_size,gamma,step,state_file=None,therm=True):
		self.nvar = wf.In + wf.Hi + wf.In*wf.Hi
		if init_state == None:
			samp = Sampler(wf,self.N_up,self.N_down)
			samp.init_state()
		else:
			wf.init_table(init_state)
			samp = Sampler(wf,self.N_up,self.N_down)

		results = Parallel(n_jobs=self.parallel_cores)(\
			delayed(get_sampler)(samp,self) for batch in range(batch_size))

		if therm == True:
			samp.thermalize(filename=state_file)

		elocals = np.array([i[0] for i in results])
		deriv_vectors = np.array([i[1] for i in results])

		forces = self.get_forces(elocals,deriv_vectors)
		Sv_operator = LinearOperator((self.nvar,self.nvar),dtype=np.complex128,\
			matvec = lambda v : self.cov_Sv(v,deriv_vectors,step))
		vec, info = cg(Sv_operator,forces)
		updates = -gamma * vec

		return updates, samp.state, np.mean(elocals)

	def get_forces(self,elocals,deriv_vectors):
		emean = np.mean(elocals)
		omean = np.mean(deriv_vectors.conj(),axis=0)
		correlator = np.mean(elocals.reshape(-1,1) * deriv_vectors.conj(),axis=0)

		return correlator - emean*omean

	def cov_Sv(self,v,deriv_vectors,step):
		# v(nvar,)
		# O = deriv_vectors(batch,nvar)
		# Ov(batch,)
		Ov = np.dot(deriv_vectors,v)
		# <O*,O>v (nvar,)
		term1 = np.dot(deriv_vectors.T.conj(), Ov) / deriv_vectors.shape[0]
		# <O*><O>v (nvar,)
		term2 = np.mean(deriv_vectors.conj(),axis=0) * np.mean(Ov)
		# reg(nvar,)
		reg = max(self.reg_list[0] * self.reg_list[1] ** step, self.reg_list[2]) * \
		        (np.mean(deriv_vectors.conj()*deriv_vectors,axis=0) - np.mean(deriv_vectors.conj(),axis=0)*\
		        	np.mean(deriv_vectors,axis=0)) * v

		return term1 - term2  + reg

	def get_deriv_vector(self,wf):
		nvar = wf.In + wf.Hi + wf.In*wf.Hi
		vector = np.zeros(nvar,dtype=np.complex128)
		vector[:wf.In] = wf.state[0]
		tanh_theta = np.tanh(wf.theta)
		vector[wf.In:wf.In+wf.Hi] = tanh_theta
		vector[wf.In+wf.Hi:] = np.dot(wf.state[0].reshape(-1,1), tanh_theta.reshape(1,-1)).ravel()

		return vector

	def get_elocal(self,wf):
		N_orbitals = wf.N_orbitals
		a = wf.a
		W = wf.W
		logpsi_2 = wf.logpsi_2
		theta = wf.theta

		def prob_1(excitation):
			occ = excitation[0] * N_orbitals + excitation[1]
			vir = excitation[0] * N_orbitals + excitation[2]
			logprob = 0 + 0j
			logprob += 2*a[vir] - 2*a[occ]
			logprob += np.sum(np.log(2*np.cosh( (theta-2*W[occ]+2*W[vir]) )) - logpsi_2)

			return np.exp(logprob)

		def prob_2(excitation):
			if excitation[0] == 2:
				# (up,down)
				occ1 = excitation[1][0]
				vir1 = excitation[2][0]
				occ2 = N_orbitals + excitation[1][1]
				vir2 = N_orbitals + excitation[2][1]
			else:
				occ1 = excitation[0] * N_orbitals + excitation[1][0]
				vir1 = excitation[0] * N_orbitals + excitation[2][0]
				occ2 = excitation[0] * N_orbitals + excitation[1][1]
				vir2 = excitation[0] * N_orbitals + excitation[2][1]
			logprob = 0 + 0j
			logprob += 2*a[vir1] + 2*a[vir2] - 2*a[occ1] - 2*a[occ2]
			logprob += np.sum(np.log(2*np.cosh( (theta - 2*W[occ1] - 2*W[occ2]	+ 2*W[vir1] + 2*W[vir2]) ))-logpsi_2)

			return np.exp(logprob)

		def sign_1(occ_orb,inx,occ,vir):
			step = 0
			if occ < vir:
				for i in range(inx+1,len(occ_orb)):
					if occ_orb[i] < vir:
						step += 1
					else:
						break
			else:
				for i in range(inx-1,-1,-1):
					if occ_orb[i] > vir:
						step += 1
					else:
						break

			return (-1)**step

		def sign_2(occ_orb,inx1,inx2,occ1,occ2,vir1,vir2):
			step = 0
			# o1,o2,v1,v2
			if vir1 > occ2:
				step += (inx2 - inx1 - 1)
				for i in range(inx2+1,len(occ_orb)):
					if occ_orb[i] > vir2:
						break
					elif occ_orb[i] < vir1:
						pass
					else:
						step += 1

			# v1,v2,o1,o2
			elif vir2 < occ1:
				step += (inx2 - inx1 - 1)
				for i in range(inx1-1,-1,-1):
					if occ_orb[i] < vir1:
						break
					elif occ_orb[i] > vir2:
						pass
					else:
						step += 1

			# v1,o1,o2,v2
			elif vir1<occ1 and occ2<vir2:
				for i in range(inx1-1,-1,-1):
					if occ_orb[i] < vir1:
						break
					else:
						step += 1
				for j in range(inx2+1,len(occ_orb)):
					if occ_orb[j] > vir2:
						break
					else:
						step +=1

			# o1,v1,o2,v2
			elif occ1<vir1 and occ2<vir2:
				for i in range(inx1+1,inx2):
					if occ_orb[i] > vir1:
						break
					else:
						step += 1
				for j in range(inx2+1,len(occ_orb)):
					if occ_orb[j] > vir2:
						break
					else:
						step +=1

			# v1,o1,v2,o2
			elif vir1<occ1 and vir2<occ2:
				for i in range(inx1-1,-1,-1):
					if occ_orb[i] < vir1:
						break
					else:
						step += 1
				for j in range(inx2-1,inx1,-1):
					if occ_orb[j] < vir2:
						break
					else:
						step += 1

			# o1,v1,v2,o2
			else:
				for i in range(inx1+1,inx2):
					if occ_orb[i] > vir1:
						break
					else:
						step += 1
				for j in range(inx2-1,inx1,-1):
					if occ_orb[j] < vir2:
						break
					else:
						step += 1


			return (-1)**step

		# now, start calsulating E_loc
		e_loc = 0.0 + 0.0j
		occ_1 = np.sort(wf.state[1]['occ_up'])
		occ_2 = np.sort(wf.state[1]['occ_down'])
		vir_1 = np.sort(wf.state[1]['vir_up'])
		vir_2 = np.sort(wf.state[1]['vir_down'])
		n_occ1 = len(occ_1)
		n_vir1 = len(vir_1)
		n_occ2 = len(occ_2)
		n_vir2 = len(vir_2)

		h1e = self.h1e
		eri = self.eri

		#<sigma|H|sigma>
		o1 = np.sum(h1e[occ_1,occ_1]) + np.sum(h1e[occ_2,occ_2])
		inx_1,inx_2 = np.meshgrid(occ_1,occ_1,indexing='ij')
		o2 = 0.5*np.sum(eri[inx_1,inx_2,inx_1,inx_2] - eri[inx_1,inx_2,inx_2,inx_1])
		inx_1,inx_2 = np.meshgrid(occ_2,occ_2,indexing='ij')
		o2 += 0.5*np.sum(eri[inx_1,inx_2,inx_1,inx_2] - eri[inx_1,inx_2,inx_2,inx_1])
		inx_1,inx_2 = np.meshgrid(occ_1,occ_2,indexing='ij')
		o2 += np.sum(eri[inx_1,inx_2,inx_1,inx_2])
		e_loc += o1+o2

		#<sigma|H|sigma_s>
		inx1,inx2,inx3,inx4 = np.meshgrid(occ_1,occ_1,vir_1,occ_1,indexing='ij')
		up_o1v1 = np.einsum('ipjp->ij',eri[inx1,inx2,inx3,inx4])
		inx1,inx2,inx3,inx4 = np.meshgrid(occ_1,occ_1,occ_1,vir_1,indexing='ij')
		up_o11v = np.einsum('ippj->ij',eri[inx1,inx2,inx3,inx4])
		inx1,inx2,inx3,inx4 = np.meshgrid(occ_1,occ_2,vir_1,occ_2,indexing='ij')
		up_o2v2 = np.einsum('ipjp->ij',eri[inx1,inx2,inx3,inx4])
		inx_1,inx_2 = np.meshgrid(occ_1,vir_1,indexing='ij')
		ov_1 = h1e[inx_1,inx_2] + up_o1v1 - up_o11v + up_o2v2
		for i in range(n_occ1):
			for j in range(n_vir1):
				occ = occ_1[i]
				vir = vir_1[j]
				e_loc += prob_1((0,occ,vir)) * sign_1(occ_1,i,occ,vir) * ov_1[i,j]
		up_o1v1,up_o11v,up_o2v2,ov_1 = (None,)*4

		inx1,inx2,inx3,inx4 = np.meshgrid(occ_2,occ_2,vir_2,occ_2,indexing='ij')
		down_o2v2 = np.einsum('ipjp->ij',eri[inx1,inx2,inx3,inx4])
		inx1,inx2,inx3,inx4 = np.meshgrid(occ_2,occ_2,occ_2,vir_2,indexing='ij')
		down_o22v = np.einsum('ippj->ij',eri[inx1,inx2,inx3,inx4])
		inx1,inx2,inx3,inx4 = np.meshgrid(occ_2,occ_1,vir_2,occ_1,indexing='ij')
		down_o1v1 = np.einsum('ipjp->ij',eri[inx1,inx2,inx3,inx4])
		inx_1,inx_2 = np.meshgrid(occ_2,vir_2,indexing='ij')
		ov_2 = h1e[inx_1,inx_2] + down_o2v2 - down_o22v + down_o1v1
		inx1,inx2,inx3,inx4,inx_1,inx_2 = (None,)*6
		for i in range(n_occ2):
			for j in range(n_vir2):
				occ = occ_2[i]
				vir = vir_2[j]
				e_loc += prob_1((1,occ,vir)) * sign_1(occ_2,i,occ,vir) * ov_2[i,j]
		down_o2v2,down_o22v,down_o1v1,ov_2 = (None,)*4

		#<sigma|H|sigma_d>
		for i in range(n_occ1):
			for j in range(i+1,n_occ1):
				for k in range(n_vir1):
					for l in range(k+1,n_vir1):
						occ1 = occ_1[i]
						occ2 = occ_1[j]
						vir1 = vir_1[k]
						vir2 = vir_1[l]
						o2 = eri[occ1,occ2,vir1,vir2] - eri[occ1,occ2,vir2,vir1]
						e_loc += prob_2((0,(occ1,occ2),(vir1,vir2))) * sign_2(occ_1,i,j,occ1,occ2,vir1,vir2) * o2

		for i in range(n_occ2):
			for j in range(i+1,n_occ2):
				for k in range(n_vir2):
					for l in range(k+1,n_vir2):
						occ1 = occ_2[i]
						occ2 = occ_2[j]
						vir1 = vir_2[k]
						vir2 = vir_2[l]
						o2 = eri[occ1,occ2,vir1,vir2] - eri[occ1,occ2,vir2,vir1]
						e_loc += prob_2((1,(occ1,occ2),(vir1,vir2))) * sign_2(occ_2,i,j,occ1,occ2,vir1,vir2) * o2

		for i in range(n_occ1):
			for j in range(n_occ2):
				for k in range(n_vir1):
					for l in range(n_vir2):
						occ1 = occ_1[i]
						occ2 = occ_2[j]
						vir1 = vir_1[k]
						vir2 = vir_2[l]
						o2 = eri[occ1,occ2,vir1,vir2]
						e_loc += prob_2((2,(occ1,occ2),(vir1,vir2))) * sign_1(occ_1,i,occ1,vir1) \
						 * sign_1(occ_2,j,occ2,vir2) * o2
		
		return e_loc


def gamma_fun(step):
	if step < 200:
		return 0.001
	elif step < 3000:
		return 0.001
	elif step < 5000:
		return 0.001
	else:
		return 1e-3

def get_sampler(mysampler,mytrainer):
	mysampler.thermalize()
	return mytrainer.get_elocal(mysampler.wf), mytrainer.get_deriv_vector(mysampler.wf)

def get_E(mysampler,mytrainer):
	mysampler.thermalize()
	return mytrainer.get_elocal(mysampler.wf)
