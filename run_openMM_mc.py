from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
import argparse
import shutil
from copy import deepcopy
import numpy as np
#******** this is module that goes with sapt force field files to generate exclusions
from sapt_exclusions import *
#***************************
#******** special NPT integrator that doesn't scale sheets
from vvintegrator5 import Barostat, Graph_list, all_atomlist, ref_atomlist

parser = argparse.ArgumentParser()
parser.add_argument("pdb", type=str, help="PDB file with initial positions")
parser.add_argument("--detail", type=str, help="Description of job")
#parser.add_argument("--devices", type=str, help="CUDADeviceIndex")
args = parser.parse_args()

if args.detail is not None:
    outPath = 'sim_' + args.detail + '/'
else:
    outPath = "output" + strftime("%s",gmtime())

if os.path.exists(outPath):
    shutil.rmtree(outPath)

temperature=300*kelvin
pressure = 1.0*atmosphere
barofreq = 200
pdb = PDBFile(args.pdb)
strdir = outPath

os.mkdir(outPath)
print(outPath)

integ_md = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
integ_md.setMaxDrudeDistance(0.02)  # this should prevent polarization catastrophe during equilibration, but shouldn't affect results afterwards ( 0.2 Angstrom displacement is very large for equil. Drudes)

pdb.topology.loadBondDefinitions('sapt_residues.xml')
pdb.topology.loadBondDefinitions('graph_residue_c.xml')
pdb.topology.loadBondDefinitions('graph_residue_s.xml')

pdb.topology.createStandardBonds();
modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField('sapt_addfunc.xml','graph_c_freeze.xml','graph_s_freeze.xml')
modeller.addExtraParticles(forcefield)
pdbresidues = [ res.name for res in pdb.topology.residues()]
if ("grpc" and "grps") in pdbresidues:
    pdbresidues = [ res.name for res in pdb.topology.residues() if (res.name != "grpc" and res.name != "grps")] 
else:
    pdbresidues = [ res.name for res in pdb.topology.residues() if res.name != "grpc"]

def removeDuplicates(listofresidues):
    reslist = []
    for res_i in listofresidues:
        if res_i not in reslist:
            reslist.append(res_i)

    return reslist

pdbresidues = removeDuplicates(pdbresidues)
print(pdbresidues)

system = forcefield.createSystem(modeller.topology, nonbondedCutoff=1.4*nanometer, constraints=None, rigidWater=True,removeCMMotion=True)
#CMMForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == CMMotionRemover][0]
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == CustomNonbondedForce][0]
nbondedForce.setNonbondedMethod(NonbondedForce.PME)
customNonbondedForce.setNonbondedMethod(min(nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))
#customNonbondedForce.setUseLongRangeCorrection(True)

# for equilibration, we want the system to be able to move in between volume moves
#CMMForce.setFrequency(1000000)

for i in range(system.getNumForces()):
    f = system.getForce(i)
    type(f)
    f.setForceGroup(i)
    # Here we are adding periodic boundaries to intra-molecular interactions.  Note that DrudeForce does not have this attribute, and
    # so if we want to use thole screening for graphite sheets we might have to implement periodic boundaries for this force type
    if type(f) == HarmonicBondForce or type(f) == HarmonicAngleForce or type(f) == PeriodicTorsionForce or type(f) == RBTorsionForce:
       f.setUsesPeriodicBoundaryConditions(True)
    f.usesPeriodicBoundaryConditions()

#barostat = MonteCarloBarostat(pressure,temperature,barofreq)
#system.addForce(barostat)
#barofreq = barostat.getFrequency()
#device_idx = args.devices
#platform = Platform.getPlatformByName('CUDA')
#properties = {'DeviceIndex': device_idx, 'Precision': 'mixed'}
#properties = {'CudaDeviceIndex': device_idx, 'CudaPrecision': 'mixed'}
platform = Platform.getPlatformByName('OpenCL')
#properties = {'OpenCLPrecision': 'single', 'OpenCLDeviceIndex':'0'}

simeq = Simulation(modeller.topology, system, integ_md, platform)
#simeq = Simulation(modeller.topology, system, integ_md, platform, properties)
simeq.context.setPositions(modeller.positions)
state = simeq.context.getState(getEnergy=True,getForces=True,getPositions=True)
position = state.getPositions()

# setup special barostat
ref_list = []
if len(pdbresidues) == 0:
    pass
elif 0 < len(pdbresidues) <= 1:
    res1 = all_atomlist(pdbresidues[0], simeq)
    res1list, Nres_res1, Natom_res1 = res1.res_list()
    res1_list = ref_atomlist(res1list, Nres_res1, Natom_res1)
    ref_list.extend(res1_list.get_ref_list())
    res2list, Nres_res2, Natom_res2 = ([], 0, 0)
    res2_list = ref_atomlist(res2list, Nres_res2, Natom_res2)
    res3list, Nres_res3, Natom_res3 = ([], 0, 0)
    res3_list = ref_atomlist(res3list, Nres_res3, Natom_res3)
elif 1 < len(pdbresidues) <= 2:
    res1 = all_atomlist(pdbresidues[0], simeq)
    res1list, Nres_res1, Natom_res1 = res1.res_list()
    res1_list = ref_atomlist(res1list, Nres_res1, Natom_res1)
    ref_list.extend(res1_list.get_ref_list())
    res2 = all_atomlist(pdbresidues[1], simeq)
    res2list, Nres_res2, Natom_res2 = res2.res_list()
    res2_list = ref_atomlist(res2list, Nres_res2, Natom_res2)
    ref_list.extend(res2_list.get_ref_list())
    res3list, Nres_res3, Natom_res3 = ([], 0, 0)
    res3_list = ref_atomlist(res3list, Nres_res3, Natom_res3)
elif 2 < len(pdbresidues) <= 3:
    res1 = all_atomlist(pdbresidues[0], simeq)
    res1list, Nres_res1, Natom_res1 = res1.res_list()
    res1_list = ref_atomlist(res1list, Nres_res1, Natom_res1)
    ref_list.extend(res1_list.get_ref_list())
    res2 = all_atomlist(pdbresidues[1], simeq)
    res2list, Nres_res2, Natom_res2 = res2.res_list()
    res2_list = ref_atomlist(res2list, Nres_res2, Natom_res2)
    ref_list.extend(res2_list.get_ref_list())
    res3 = all_atomlist(pdbresidues[2], simeq)
    res3list, Nres_res3, Natom_res3 = res3.res_list()
    res3_list = ref_atomlist(res3list, Nres_res3, Natom_res3)
    ref_list.extend(res3_list.get_ref_list())

merge_ref = np.ravel(ref_list).tolist()

veclist = []
if len(pdbresidues) == 0:
    veclist.extend(res1_list.get_vectors(position))
elif 1 < len(pdbresidues) <= 2:
    veclist.extend(res1_list.get_vectors(position))
    veclist.extend(res2_list.get_vectors(position))
elif 2 < len(pdbresidues) <= 3:
    veclist.extend(res1_list.get_vectors(position))
    veclist.extend(res2_list.get_vectors(position))
    veclist.extend(res3_list.get_vectors(position))

#if len(res1_list.get_vectors(position)) == 0:
#    pass
#else:
#    veclist.extend(res1_list.get_vectors(position))
##    print(catlist.get_vectors(position), np.shape(catlist.get_vectors(position)))
#
#if len(res2_list.get_vectors(position)) == 0:
#    pass
#else:
#    veclist.extend(res2_list.get_vectors(position))
##    print(anlist.get_vectors(position), np.shape(anlist.get_vectors(position)))
#
#if len(res3_list.get_vectors(position)) == 0:
##if len(pdbresidues) <= 2:
#    pass
#else:
#    veclist.extend(res3_list.get_vectors(position))

pdbresidues = [ res.name for res in pdb.topology.residues()]
if ("grpc" and "grps") in pdbresidues:
    grpc = Graph_list("grpc", simeq)
    first_idx, second_idx = grpc.grpclist()
    grps = Graph_list("grps", simeq)
    grps.grpclist()
    cathode_idx = int(first_idx + len(grps.steps)/2)
    print(cathode_idx, second_idx, cathode_idx)
else:
    grpc = Graph_list("grpc", simeq)
    first_idx, second_idx = grpc.grpclist()
    cathode_idx = int(first_idx)
    print(cathode_idx, second_idx, cathode_idx)
#first_idx=3203
#second_idx=6406
#barostat = Barostat(simeq,pressure,temperature,barofreq,first_idx+1,second_idx+1, 0.05)
barostat = Barostat(simeq,pressure,temperature,barofreq,first_idx+1,second_idx+1, cathode_idx+1, 0.05)

#************************************************
#         IMPORTANT: generate exclusions for SAPT-FF
#
sapt_exclusions = sapt_generate_exclusions(simeq,system,modeller.positions)
#
#************************************************

# initial energies
state = simeq.context.getState(getEnergy=True,getForces=True,getPositions=True)
position = state.getPositions()

print(str(state.getKineticEnergy()))
print(str(state.getPotentialEnergy()))
for j in range(system.getNumForces()):
    f = system.getForce(j)
    print(type(f), str(simeq.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
# write initial pdb file with drude oscillators
PDBFile.writeFile(simeq.topology, position, open(strdir+'start_drudes.pdb', 'w'))


simeq.reporters = []
simeq.reporters.append(DCDReporter(strdir+'md_npt.dcd', 1000))
simeq.reporters.append(CheckpointReporter(strdir+'md_npt.chk', 10000))
simeq.reporters[1].report(simeq,state)

print('Simulating...')

for i in range(1,10000):
    #barostat.step(1000, merge_ref)
    barostat.step(1000, merge_ref, veclist, system, res1_list, res2_list, res3_list, position)
    #barostat.step(1000)
    print(i,strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print(i,datetime.now())
    state = simeq.context.getState(getEnergy=True,getForces=True,getPositions=True)
    print(str(state.getKineticEnergy()))
    print(str(state.getPotentialEnergy()))
    for j in range(system.getNumForces()):
        f = system.getForce(j)
        print(type(f), str(simeq.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

print('Done!')

exit()
