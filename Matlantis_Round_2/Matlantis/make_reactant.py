from ase import Atoms, io

o_atom = Atoms('O', positions=[[0, 0, 0]])
io.write('Matlantis/reactant.traj', o_atom)
print('Written reactant.traj')
