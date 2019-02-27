#!/usr/bin/env python

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
from copy import deepcopy
import os
import sys
import numpy
import argparse
import shutil

from subroutines import *


parser = argparse.ArgumentParser()
parser.add_argument("pdb", type=str, help="PDB file with initial positions")
parser.add_argument("--nstep", type=str, help="number of step for updating forces")
parser.add_argument("--volt", type=str, help="applied potential (V)")
parser.add_argument("--nsec", type=str, help="simulation time (ns)")
parser.add_argument("--devices", type=str, help="CUDADeviceIndex")
parser.add_argument("--temp", type=str, help="temperature (K)")
#parser.add_argument("--devices", type=str, nargs="*", default=[])
args = parser.parse_args()

if args.nstep is not None:
    outPath = 'simmd_' + args.nstep + "step_" + args.volt + "V_" + args.nsec + "ns_" + args.temp + 'K'
else:
    outPath = "output" + strftime("%s",gmtime())

if os.path.exists(outPath):
    shutil.rmtree(outPath)

strdir ='../'
os.mkdir(outPath)
os.chdir(outPath)
chargesFile = open("charges.dat", "w")
print(outPath)

pdb = args.pdb
device_idx = args.devices
temperature = float(args.temp)
sim = MDsimulation( 
        pdb, 
        device_idx,
        temperature,
        ResidueConnectivityFiles = [
                #strdir + 'ffdir/sapt_chlf.xml', 
                strdir + 'ffdir/sapt_residues.xml', 
                strdir + 'ffdir/graph_residue_c.xml',
                strdir + 'ffdir/graph_residue_n.xml'
        ],
        FF_files = [
                #strdir + 'ffdir/sapt_noDB.xml', 
                #strdir + 'ffdir/sapt_chlf_2sheets.xml', 
                strdir + 'ffdir/sapt_noDB_2sheets.xml', 
                strdir + 'ffdir/graph_c_freeze.xml',
                strdir + 'ffdir/graph_n_freeze.xml'
        ]
        #FF_Efield_files = [
        #        strdir + 'ffdir/sapt_Efield_noDB.xml', 
        #        strdir + 'ffdir/graph_c_freeze.xml'
        #        strdir + 'ffdir/graph_n_freeze.xml'
        #]
)

sim.equilibration()

print('Starting Production NPT Simulation...')

# interior sheets which have fluctuation charges are now labeled "grpc", and sheets which remain neutral are labeled
# "grph"
grpc = Graph_list("grpc")
grpc.grpclist(sim)
grp_dummy = Graph_list("grpd")
grp_dummy.grpclist(sim)
grp_n = Graph_list("grph")
grp_n.grpclist(sim)

graph = deepcopy(grpc.cathode)
graph.extend(deepcopy(grp_dummy.dummy[:int(len(grp_dummy.dummy)/2)]))
graph.extend(deepcopy(grpc.anode))
graph.extend(deepcopy(grp_dummy.dummy[int(len(grp_dummy.dummy)/2): len(grp_dummy.dummy)]))
assert len(graph) == 2400
grph = deepcopy(grp_n.neutral)

## H atoms of solution
#HofBMI = solution_Hlist("BMIM")
#HofBMI.cation_hlist(sim)
#BofBF4 = solution_Hlist("BF4")
#BofBF4.anion_hlist(sim)
#HofACN = solution_Hlist("acnt")
#HofACN.solvent_hlist(sim)
#He = solution_Hlist("Hel")
#He.vac_list(sim)
#
#merge_Hlist= deepcopy(HofBMI.cation)
#merge_Hlist.extend( deepcopy(BofBF4.anion) )
#merge_Hlist.extend( deepcopy(HofACN.solvent) )
#He_list = deepcopy(He.He)

# all atoms of solution
pdbresidues = [ res.name for res in sim.simmd.topology.residues()]
if ("grpc" and "grpd" and "grph" and "grps") in pdbresidues:
    pdbresidues = [ res.name for res in sim.simmd.topology.residues() if (res.name != "grpc" and res.name != "grpd" and res.name != "grph" and res.name != "grps")]
elif ("grpc" and "grpd" and "grph") in pdbresidues:
    pdbresidues = [ res.name for res in sim.simmd.topology.residues() if (res.name != "grpc" and res.name != "grpd" and res.name != "grph")]
elif ("grpc" and "grpd") in pdbresidues:
    pdbresidues = [ res.name for res in sim.simmd.topology.residues() if (res.name != "grpc" and res.name != "grpd")]
elif ("grpc" and "grps") in pdbresidues:
    pdbresidues = [ res.name for res in sim.simmd.topology.residues() if (res.name != "grpc" and res.name != "grps")]
else:
    pdbresidues = [ res.name for res in sim.simmd.topology.residues() if res.name != "grpc"]

def removeDuplicates(listofresidues):
    reslist = []
    for res_i in listofresidues:
        if res_i not in reslist:
            reslist.append(res_i)

    return reslist

pdbresidues = removeDuplicates(pdbresidues)
print(pdbresidues)
if len(pdbresidues) == 0:
    pass
elif 0 < len(pdbresidues) <= 1:
    res1 = solution_allatom(pdbresidues[0])
    res1.res_list(sim)
    solvent_list = deepcopy(res1.atomlist)
elif 1 < len(pdbresidues) <= 2:
    res1 = solution_allatom(pdbresidues[0])
    res1.res_list(sim)
    res2 = solution_allatom(pdbresidues[1])
    res2.res_list(sim)
    solvent_list = deepcopy(res1.atomlist)
    solvent_list.extend(deepcopy(res2.atomlist))
elif 2 < len(pdbresidues) <= 3:
    res1 = solution_allatom(pdbresidues[0])
    res1.res_list(sim)
    res2 = solution_allatom(pdbresidues[1])
    res2.res_list(sim)
    res3 = solution_allatom(pdbresidues[2])
    res3.res_list(sim)
    solvent_list = deepcopy(res1.atomlist)
    solvent_list.extend(deepcopy(res2.atomlist))
    solvent_list.extend(deepcopy(res3.atomlist))

# add exclusions for intra-sheet non-bonded interactions.
sim.exlusionNonbondedForce(graph, grph)
state = sim.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True,getParameters=True)
initialPositions = state.getPositions()
#cell_dist = Distance(grpc.c562_1, grpc.c562_2, initialPositions)
cell_dist, z_L, z_R = Distance(grpc.c562_1, grpc.c562_2, initialPositions)
print(z_L, z_R)
print('cathode-anode distance (nm)', cell_dist)

boxVecs = sim.simmd.topology.getPeriodicBoxVectors()
crossBox = numpy.cross(boxVecs[0], boxVecs[1])
sheet_area = numpy.dot(crossBox, crossBox)**0.5 / nanometer**2
print(sheet_area)

sim.simmd.context.reinitialize()
#sim.simEfield.context.reinitialize()
sim.simmd.context.setPositions(initialPositions)
#sim.simEfield.context.setPositions(initialPositions)


#************ get rid of the MD loop, just calculating converged charges ***********
Ngraphene_atoms = len(graph)

# one sheet here
area_atom = sheet_area / (Ngraphene_atoms / 2) # this is the surface area per graphene atom in nm^2
conv = 18.8973 / 2625.5  # bohr/nm * au/(kJ/mol)
# z box coordinate (nm)
zbox=boxVecs[2][2] / nanometer
Lgap = (zbox - cell_dist) # length of vacuum gap in nanometers, set by choice of simulation box (z-dimension)
print('length of vacuum gap (nm)', Lgap)
Niter_max = 3  # maximum steps for convergence
tol=0.01 # tolerance for average deviation of charges between iteration
Voltage = float(args.volt)  # external voltage in Volts
Voltage = Voltage * 96.487  # convert eV to kJ/mol to work in kJ/mol
q_max = 2.0  # Don't allow charges bigger than this, no physical reason why they should be this big
f_iter = int(( float(args.nsec) * 1000000 / int(args.nstep) )) + 1  # number of iterations for charge equilibration
#print('number of iterations', f_iter)
small = 1e-4

sim.initializeCharge( Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, small, cell_dist)

allEz_cell = []
allEx_i = []
allEy_i = []
allEz_i = []
for i in range(1, f_iter ):
    print()
    print(i,datetime.now())

    sim.simmd.step( int(args.nstep) )

    state = sim.simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
    print(str(state.getKineticEnergy()))
    print(str(state.getPotentialEnergy()))

    positions = state.getPositions()
#    sim.simmd.context.setPositions(positions)
#    sim.simEfield.context.setPositions(positions)
#    sim.nbondedForce_Efield.updateParametersInContext(sim.simEfield.context)
#    sim.nbondedForce.updateParametersInContext(sim.simmd.context)
    sim.ConvergedCharge( Niter_max, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, q_max )
    sumq_cathode, sumq_anode = sim.FinalCharge(Ngraphene_atoms, graph, args, i, chargesFile)
    print( 'total charge on graphene (cathode,anode):', sumq_cathode, sumq_anode )
    print('Charges converged, Energies from full Force Field')
    sim.PrintFinalEnergies()

    ind_Q = get_Efield(solvent_list)
    ana_Q_Cat, ana_Q_An = ind_Q.induced_q( z_L, z_R, cell_dist, sim, positions, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv)
    print('Analytical Q_Cat, Q_An :', ana_Q_Cat, ana_Q_An)
    
    sim.Scale_charge( Ngraphene_atoms, graph, ana_Q_Cat, ana_Q_An, sumq_cathode, sumq_anode)
    #state2 = sim.simEfield.context.getState(getEnergy=True,getForces=True,getPositions=True)
    #forces = state2.getForces()
#    state = sim.simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
#    forces = state.getForces()
    

print('Done!')

exit()
