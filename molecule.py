import numpy as np
from pyscf import gto, scf, ao2mo
from functools import reduce

def rebuild(eri,N_orbitals,spin):
	new_eri = np.zeros((N_orbitals,)*4)
	for i in range(N_orbitals):
		for j in range(N_orbitals):
			for k in range(N_orbitals):
				for l in range(N_orbitals):
					# <ij|kl> = (ik|jl)
					linx = [i,k]
					rinx = [j,l]
					linx.sort()
					rinx.sort()
					i0, k0 = linx
					j0, l0 = rinx
					new_eri[i,j,k,l] = eri[k0*(k0+1)//2+i0,l0*(l0+1)//2+j0]

	return new_eri

def get_mol_info(mol,geo_opt=False):
	myhf = scf.RHF(mol).run()

	if geo_opt:
		from pyscf.geomopt.berny_solver import optimize
		mol = optimize(myhf)
		myhf = scf.RHF(mol).run()

	mo = myhf.mo_coeff
	N_orbitals = mo.shape[-1]
	N_electrons = mol.nelectron
	N_up = (N_electrons + mol.spin) // 2
	N_down = (N_electrons - mol.spin) // 2 

	h1e = reduce(np.dot,(mo.T,scf.hf.get_hcore(mol),mo))
	eri = ao2mo.full(mol,mo)
	
	eri = rebuild(eri,N_orbitals,mol.spin)

	return h1e, eri, N_orbitals, N_up, N_down, mol.energy_nuc()

'''
def training(mol,geo_opt=False,load_file=None,alpha=2,state_file=None,param_file=None,cores=1):
	
	# mol: the molecule to calculate
	# geo_opt: boolean. Whether do geometry optimization, default False. And default method is HF
	# load_file: pre-optimized RBM parameters
	# state_file: write states during training
	# param_file: save RBM parameters
	
	h1e, eri, N_orbitals, N_up, N_down, ecore = get_mol_info(mol,geo_opt=False)

	wf = Nqs(N_orbitals,alpha)
	# Initiate RBM parameters
	if load_file == None:
		wf.W = 0.1 * np.random.random(wf.W.shape) + 0.1j * np.random.random(wf.W.shape)
		wf.a = 0.1 * np.random.random(wf.a.shape) + 0.1j * np.random.random(wf.a.shape)
		wf.b = 0.1 * np.random.random(wf.b.shape) + 0.1j * np.random.random(wf.b.shape)
	else:
		wf.load_parameters(load_file)

	mytrainer = Trainer((h1e,eri,N_up,N_down),cores=cores)
	wf, elist = mytrainer.train(wf,ecore,batch_size=100,num_steps=5000,init_state=None,\
		print_freq=20,state_file=state_file,param_file= param_file,out_freq=50)

	E = mytrainer.eval(wf,ecore)
	print('Ground State Energy: ',E)

'''
