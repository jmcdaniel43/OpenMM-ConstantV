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

if len(sys.argv) > 1:
    outPath = 'output_' +  sys.argv[1]
else:
    outPath = "output" + strftime("%s",gmtime())

temperature=300*kelvin
cutoff=1.4*nanometer

strdir='../'
os.mkdir(outPath)
os.chdir(outPath)
chargesFile = open("charges.dat", "w")
print(outPath)

pdb = PDBFile(strdir+'equil_nvt_noDrudes.pdb')

integ_md = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
integ_md.setMaxDrudeDistance(0.02)  # this should prevent polarization catastrophe during equilibration, but shouldn't affect results afterwards ( 0.2 Angstrom displacement is very large for equil. Drudes)
# this integrator won't be used, its just for Efield electric field calculation
integ_junk = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)

pdb.topology.loadBondDefinitions(strdir+'sapt_residues.xml')
pdb.topology.loadBondDefinitions(strdir+'graph_residue_c.xml')
pdb.topology.loadBondDefinitions(strdir+'graph_residue_n.xml')
pdb.topology.createStandardBonds()

modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField(strdir+'sapt.xml',strdir+'graph_c.xml',strdir+'graph_n.xml')
modeller.addExtraParticles(forcefield)

# this force field has only Coulomb interactions which is used to compute electric field
modeller2 = Modeller(pdb.topology, pdb.positions)
forcefield_Efield = ForceField(strdir+'sapt_Efield.xml',strdir+'graph_c.xml',strdir+'graph_n.xml')
modeller2.addExtraParticles(forcefield_Efield)

system = forcefield.createSystem(modeller.topology, nonbondedCutoff=cutoff, constraints=None, rigidWater=True)
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == CustomNonbondedForce][0]
drudeForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == DrudeForce][0]
nbondedForce.setNonbondedMethod(NonbondedForce.PME)
customNonbondedForce.setNonbondedMethod(min(nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))
print('nbMethod : ', customNonbondedForce.getNonbondedMethod())

for i in range(system.getNumForces()):
    f = system.getForce(i)
    type(f)
    f.setForceGroup(i)
    # Here we are adding periodic boundaries to intra-molecular interactions.  Note that DrudeForce does not have this attribute, and
    # so if we want to use thole screening for graphite sheets we might have to implement periodic boundaries for this force type
    if type(f) == HarmonicBondForce or type(f) == HarmonicAngleForce or type(f) == PeriodicTorsionForce or type(f) == RBTorsionForce:
       f.setUsesPeriodicBoundaryConditions(True)
    f.usesPeriodicBoundaryConditions()

# set up system2 for Efield calculation
system_Efield = forcefield_Efield.createSystem(modeller2.topology, nonbondedCutoff=cutoff, constraints=None, rigidWater=True)
nbondedForce_Efield = [f for f in [system_Efield.getForce(i) for i in range(system_Efield.getNumForces())] if type(f) == NonbondedForce][0]
nbondedForce_Efield.setNonbondedMethod(NonbondedForce.PME)


totmass = 0.*dalton
for i in range(system.getNumParticles()):
    totmass += system.getParticleMass(i)


#platform = Platform.getPlatformByName('CPU')
platform = Platform.getPlatformByName('OpenCL')
properties = {'OpenCLPrecision': 'mixed', 'OpenCLDeviceIndex': '0'}

simmd = Simulation(modeller.topology, system, integ_md, platform)
simmd.context.setPositions(modeller.positions)

# set up simulation for Efield
simEfield = Simulation(modeller2.topology, system_Efield, integ_junk, platform)

platform = simmd.context.getPlatform()
platformname = platform.getName();
print(platformname)

# Initialize energy
state = simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
print('Initial Energy')
print(str(state.getKineticEnergy()))
print(str(state.getPotentialEnergy()))
for j in range(system.getNumForces()):
    f = system.getForce(j)
    print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

t1 = datetime.now()
# write initial pdb with drude oscillators
position = state.getPositions()
simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
PDBFile.writeFile(simmd.topology, position, open('start_drudes.pdb', 'w'))

simmd.reporters = []
simmd.reporters.append(DCDReporter('md_nvt.dcd', 1000))
simmd.reporters.append(CheckpointReporter('md_nvt.chk', 10000))
simmd.reporters[1].report(simmd,state)

# print('Equilibrating...')
# for i in range(5000):
#     simmd.step(1000)

state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True,enforcePeriodicBox=True)
position = state.getPositions()
simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
# PDBFile.writeFile(simmd.topology, position, open(strdir+'equil_nvt.pdb', 'w'))

state = simmd.context.getState(getPositions=True)
initialPositions = state.getPositions()
simmd.context.reinitialize()
simmd.context.setPositions(initialPositions)

state = simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True,getParameters=True)
print('Equilibrated Energy')
print(str(state.getKineticEnergy()))
print(str(state.getPotentialEnergy()))
for j in range(system.getNumForces()):
    f = system.getForce(j)
    print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

print('Starting Production NPT Simulation...')

# interior sheets which have fluctuation charges are now labeled "grpc", and sheets which remain neutral are labeled
# "grph"
res_idx = -1
c562_1 = -1
c562_2 = -1
cathode = []
anode = []
for res in simmd.topology.residues():
    if res.name == "grpc":
        for atom in res._atoms:
            (q_i, sig, eps) = nbondedForce.getParticleParameters(atom.index)
            if q_i._value != 0:
                print(atom, q_i)
        if res_idx == -1:
            res_idx = res.index
            for atom in res._atoms:
                cathode.insert(int(atom.name[1:]), atom.index)
                if atom.name == "C562":
                    c562_1 = atom.index
        elif res.index != res_idx:
            for atom in res._atoms:
                anode.insert(int(atom.name[1:]), atom.index)
                if atom.name == "C562":
                    c562_2 = atom.index

graph = deepcopy(cathode)
graph.extend(deepcopy(anode))
assert len(graph) == 1600


#******* JGM ****************
# add exclusions for intra-sheet non-bonded interactions.

# first figure out which exclusions we already have (1-4 atoms and closer).  The code doesn't
# allow the same exclusion to be added twice
flagexclusions = {}
for i in range(customNonbondedForce.getNumExclusions()):
    (particle1, particle2) = customNonbondedForce.getExclusionParticles(i)
    string1=str(particle1)+"_"+str(particle2)
    string2=str(particle2)+"_"+str(particle1)
    flagexclusions[string1]=1
    flagexclusions[string2]=1

# now add exclusions for every atom pair in each sheet if we don't already have them
#cathode first.
for i in range(len(cathode)):
    indexi = cathode[i]
    for j in range(i+1,len(cathode)):
        indexj = cathode[j]
        string1=str(indexi)+"_"+str(indexj)
        string2=str(indexj)+"_"+str(indexi)
        if string1 in flagexclusions and string2 in flagexclusions:
            continue
        else:
            customNonbondedForce.addExclusion(indexi,indexj)
            nbondedForce.addException(indexi,indexj,0,1,0,True)
            nbondedForce_Efield.addException(indexi,indexj,0,1,0,True)
#now anode
for i in range(len(anode)):
    indexi = anode[i]
    for j in range(i+1,len(anode)):
        indexj = anode[j]
        string1=str(indexi)+"_"+str(indexj)
        string2=str(indexj)+"_"+str(indexi)
        if string1 in flagexclusions and string2 in flagexclusions:
            continue
        else:
            customNonbondedForce.addExclusion(indexi,indexj)
            nbondedForce.addException(indexi,indexj,0,1,0,True)
            nbondedForce_Efield.addException(indexi,indexj,0,1,0,True)


initialPositions = state.getPositions()

pos_c562_1 = initialPositions[c562_1]
pos_c562_2 = initialPositions[c562_2]
cell_dist = 0
for i in range(3):
    d = pos_c562_1[i] / nanometer - pos_c562_2[i] / nanometer
    cell_dist += (d**2)

cell_dist = cell_dist**(1/2)

print('cathode-anode distance (nm)', cell_dist)

boxVecs = simmd.topology.getPeriodicBoxVectors()
crossBox = numpy.cross(boxVecs[0], boxVecs[1])
sheet_area = numpy.dot(crossBox, crossBox)**0.5 / nanometer**2
print(sheet_area)

simmd.context.reinitialize()
simEfield.context.reinitialize()
simmd.context.setPositions(modeller.positions)
simEfield.context.setPositions(modeller.positions)


#************ get rid of the MD loop, just calculating converged charges ***********
Ngraphene_atoms = len(graph)

# one sheet here
area_atom = sheet_area / (Ngraphene_atoms / 2) # this is the surface area per graphene atom in nm^2
conv = 18.8973 / 2625.5  # bohr/nm * au/(kJ/mol)
# z box coordinate (nm)
zbox=boxVecs[2][2] / nanometer
Lgap = (zbox - cell_dist) # length of vacuum gap in nanometers, set by choice of simulation box (z-dimension)
print('length of vacuum gap (nm)', Lgap)
Niter_max = 10  # maximum steps for convergence
tol=0.01 # tolerance for average deviation of charges between iteration
Voltage = 2.0  # external voltage in Volts
Voltage = Voltage * 96.487  # convert eV to kJ/mol to work in kJ/mol
q_max = 2.0  # Don't allow charges bigger than this, no physical reason why they should be this big

for i in range(1,50001):
    print()
    print(i,datetime.now())

    simmd.step(100)

    state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
    print(str(state.getKineticEnergy()))
    print(str(state.getPotentialEnergy()))

    for j in range(system.getNumForces()):
        f = system.getForce(j)
        print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

    positions = state.getPositions()
    simEfield.context.setPositions(positions)

    rms = 0.0
    flag_conv = -1
    for i_step in range(Niter_max):
        print("charge iteration", i_step)

        state2 = simEfield.context.getState(getEnergy=True,getForces=True,getPositions=True)
        for j in range(system_Efield.getNumForces()):
                f = system_Efield.getForce(j)
                print(type(f), str(simEfield.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

        forces = state2.getForces()
        for i_atom in range(Ngraphene_atoms):
                index = graph[i_atom]
                (q_i_old, sig, eps) = nbondedForce_Efield.getParticleParameters(index)
                q_i_old = q_i_old
                E_z = forces[index][2]._value / q_i_old._value
                E_i_external = E_z

                # when we switch to atomic units on the right, sigma/2*epsilon0 becomes 4*pi*sigma/2 , since 4*pi*epsilon0=1 in a.u.
                if i_atom < Ngraphene_atoms / 2:
                    q_i = 2.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + E_i_external) * conv
                else:  # anode
                    q_i = -2.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + E_i_external) * conv

                # Make sure calculated charge isn't crazy
                if abs(q_i) > q_max:
                    # this shouldn't happen.  If this code is run, we might have a problem
                    # for now, just don't use this new charge
                    q_i = q_i_old._value
                    print("ERROR: q_i > q_max: {:f} > {:f}".format(q_i, q_max))

                nbondedForce_Efield.setParticleParameters(index, q_i, sig, eps)
                rms += (q_i - q_i_old._value)**2

        nbondedForce_Efield.updateParametersInContext(simEfield.context)

        rms = (rms/Ngraphene_atoms)**0.5
        if rms < tol:
            flag_conv = i_step
            break

    # warn if not converged
    if flag_conv == -1:
        print("Warning:  Electrode charges did not converge!! rms: %f" % (rms))
    else:
        print("Steps to converge: " + str(flag_conv + 1))

    sumq_cathode=0
    sumq_anode=0
    print('Final charges on graphene atoms')
    for i_atom in range(Ngraphene_atoms):
        index = graph[i_atom]
        (q_i, sig, eps) = nbondedForce_Efield.getParticleParameters(index)
        nbondedForce.setParticleParameters(index, q_i, 1.0 , 0.0)

        # if we are on a 1000 step interval, write charges to file
        # i starts at 0, so at i = 9, 1000 frames will have occured
        if i % 10 == 0:
            chargesFile.write("{:f} ".format(q_i._value))

        if i_atom < Ngraphene_atoms / 2:
            # print charge on one representative atom for debugging fluctuations
            if i_atom == 100:
                print('index, charge, sum',index, q_i , sumq_cathode ) 
            sumq_cathode += q_i._value
        else:
            # print charge on one representative atom for debugging fluctuations
            if i_atom == Ngraphene_atoms/2 + 100:
                print('index, charge, sum',index, q_i, sumq_anode )
            sumq_anode += q_i._value

    # write newline to charge file after charge write
    if i % 10 == 0:
        chargesFile.write("\n")

    print( 'total charge on graphene (cathode,anode):', sumq_cathode, sumq_anode )

    print('Charges converged, Energies from full Force Field')
    nbondedForce.updateParametersInContext(simmd.context)

    state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
    for j in range(system.getNumForces()):
        f = system.getForce(j)
        print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

print('Done!')

exit()
