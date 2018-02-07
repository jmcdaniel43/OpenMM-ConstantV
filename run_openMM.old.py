from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from time import gmtime, strftime
from datetime import datetime

temperature=300*kelvin
cutoff=1.4*nanometer

pdb = PDBFile('system.pdb')
strdir = ''

integ_md = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
integ_md.setMaxDrudeDistance(0.02)  # this should prevent polarization catastrophe during equilibration, but shouldn't affect results afterwards ( 0.2 Angstrom displacement is very large for equil. Drudes)
# this integrator won't be used, its just for Efield electric field calculation
integ_junk = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)


pdb.topology.loadBondDefinitions('sapt_residues.xml')
pdb.topology.loadBondDefinitions('graph_residue_c.xml')
pdb.topology.loadBondDefinitions('graph_residue_n.xml')
pdb.topology.createStandardBonds();

modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField('sapt.xml','graph_c.xml','graph_n.xml')
modeller.addExtraParticles(forcefield)

# this force field has only Coulomb interactions which is used to compute electric field
modeller2 = Modeller(pdb.topology, pdb.positions)
forcefield_Efield = ForceField('sapt_Efield.xml','graph_c.xml','graph_n.xml')
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


#platform = Platform.getPlatformByName('CUDA')
platform = Platform.getPlatformByName('CPU')
#properties = {'CudaPrecision': 'mixed'}

simmd = Simulation(modeller.topology, system, integ_md, platform)
simmd.context.setPositions(modeller.positions)

# set up simulation for Efield
simEfield = Simulation(modeller2.topology, system_Efield, integ_junk, platform)


platform =simmd.context.getPlatform()
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

# minimization screws up velocities, not sure why.  This is probably a bug in OpenMM
#print('Minimizing Energy...')
#simmd.minimizeEnergy(maxIterations=2000)
#print('Minimization finished !')

simmd.loadCheckpoint('md_npt.chk')

print('Starting Production NPT Simulation...')
t1 = datetime.now()
# write initial pdb with drude oscillators
position = state.getPositions()
simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
PDBFile.writeFile(simmd.topology, position, open(strdir+'start_drudes.pdb', 'w'))
simmd.reporters = []
simmd.reporters.append(DCDReporter(strdir+'md_npt2.dcd', 1000))
simmd.reporters.append(CheckpointReporter(strdir+'md_npt2.chk', 10000))
simmd.reporters[1].report(simmd,state)

# Corners of the graphene sheet are atoms C562, C460, C461
# Measuring the distance between atoms C562 in adjacent sheets will give the length
# of the gap of the capacitor.
# [ 120 (the size of the z dimension for the peridoc box) - dist(C562_1, C562_2) ] / 2
# will give the length of the gap between sheets.
# Area of the sheet can be calculated by dist(C562, C461) * dist(C461, C462)

# indicies of the relevant atoms
c562_1 = -1
c562_2 = -1 # C562 atom in the second graphene sheet
c461 = -1
c460 = -1

# Find relevant C atoms
cathode_idx = -1
anode_idx = -1
cathode = []
anode = []
graph = []

# JGM changed residue names so this code should still work with multiple graphene sheets per electrode
# interior sheets which have fluctuation charges are now labeled "grpc", and sheets which remain neutral are labeled
# "grph"
for res in simmd.topology.residues():
  if res.name == "grpc":  #modified JGM
    if cathode_idx == -1:
        # we only want to consider atoms from one sheet of the graphene
        # so force one residue index for this loop
        cathode_idx = res.index
    elif res.index != cathode_idx:
        continue

    for atom in res._atoms:
        cathode.insert(int(atom.name[1:]), atom.index)

        if atom.name in ["C562", "C461", "C460"]:
            if atom.name == "C562":
                c562_1 = atom.index
            elif atom.name == "C461":
                c461 = atom.index
            elif atom.name == "C460":
                c460 = atom.index

for res in simmd.topology.residues():
    if res.name == "grpc":  #modified JGM
        ## now we'll look at the other sheet to get the other C562 atom
        if cathode_idx == res.index:
            continue

        anode_idx = res.index

        for atom in res._atoms:
            anode.insert(int(atom.name[1:]), atom.index)

            if atom.name == "C562":
                c562_2 = atom.index

graph = cathode
graph.extend(anode)

initialPositions = state.getPositions()

pos_c562_1 = initialPositions[c562_1]
pos_c562_2 = initialPositions[c562_2]
cell_dist = 0
for i in range(3):
    d = pos_c562_1[i] / nanometer - pos_c562_2[i] / nanometer
    cell_dist += (d**2)
cell_dist = cell_dist**(1/2)

pos_c461 = initialPositions[c461]
pos_c460 = initialPositions[c460]
sheet_length = 0
sheet_height = 0
for i in range(3):
    dl = pos_c562_1[i] / nanometer - pos_c461[i] / nanometer
    dh = pos_c461[i] / nanometer - pos_c460[i] / nanometer
    sheet_length += dl**2
    sheet_height += dh**2
sheet_length = sheet_length**(1/2)
sheet_height = sheet_height**(1/2)
sheet_area = sheet_length * sheet_height

#**** JGM, not sure if sheet_area is correct.  should be 48.651**2 * cos(30) since unit cell is not rectangular
print('Simulating...')

for i in range(1,50001):
    simmd.step(100)
    print(i,strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print(i,datetime.now())
    state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
    print(str(state.getKineticEnergy()))
    print(str(state.getPotentialEnergy()))

    #********************** Code goes here for fixing electrode charges to constant potential *************************************
    #  for testing, we can take this outside of loop, but eventually we will run this for every step ( or even N steps ) in the MD simulation

    # store positions for Efield calculation
    position = state.getPositions()
    simEfield.context.setPositions(position)

    # here's the pseudo code
    Ngraphene_atoms = 1600
    q_max = 2.0  # Don't allow charges bigger than this, no physical reason why they should be this big
    tol=0.01 # tolerance for average deviation of charges between iteration
    Niter_max = 20  # maximum steps for convergence
    Lgap = (120 - cell_dist) / 2.0  # length of vacuum gap in nanometers, set by choice of simulation box (z-dimension)
    Voltage = 2  # external voltage in Volts
    Voltage = Voltage * 96.487  # convert eV to kJ/mol to work in kJ/mol
    area_atom = sheet_area / Ngraphene_atoms / 2 # this is the surface area per graphene atom in nm^2
    conv = 18.8973 / 2625.5  # bohr/nm * au/(kJ/mol)
    # iterate until convergence
    flag_conv=-1
    for i_step in range(Niter_max):

        # for convergence test
        rms=0.0

        # First, get electric fields on all atoms using OpenMM libraries, keeping positions fixed
        # We have to do this for each iteration, this is why this is a costly method
        state2 = simEfield.context.getState(getEnergy=True,getForces=True,getPositions=True)
        forces = state2.getForces()
        # because the forces are purely Coulombic from this context, we easily get electric field from E=F/q

        # loop over all graphene atoms.  Tecnically, we should only loop over atoms on outer sheet, since there can only be charge on the surface of a conductor
        # for the initial implementation, we should use electrodes which only consist of a single graphene sheet.
        for i_atom in range(Ngraphene_atoms):
            index = graph[i_atom]  # this is global index of atom for OpenMM data structures
            # current charge on atom
            (q_i_old, sig, eps) = nbondedForce_Efield.getParticleParameters(index)
            q_i_old = q_i_old
            # external electric field on atom, here we only use Ez
            # in principle, Ex, Ey should equal zero for graphene atoms.  This should fall out naturally, but should be verified.
            E_z = forces[index][2]._value / q_i_old._value  # this will be a problem if charges are zero... however if we start from finite charges hopefully they will never be numerically zero
            E_i_external = E_z
            # new charge satisfying Gauss's Law at Electrode surface
            # we have to be careful of units here.  force is output from openMM in kJ/mol/nm , easiest to use external potential in kJ/mol, see conversion above

            # set one graphite sheet as cathode, one as anode
            # assume cathode is on the left, consistent with sign of E_z
            if i_atom < Ngraphene_atoms / 2:
                q_i = 2.0 * area_atom * ( Voltage / Lgap + E_i_external ) * conv
            else:  # anode
                q_i = -2.0 * area_atom * ( Voltage / Lgap + E_i_external ) * conv

            # Make sure calculated charge isn't crazy
            if abs(q_i) > q_max:
                # this shouldn't happen.  If this code is run, we might have a problem
                # for now, just don't use this new charge
                q_i = q_i_old._value

            rms+=(q_i - q_i_old._value)**2

            # update charges only in Efield system for now
            nbondedForce_Efield.setParticleParameters(index, q_i, sig, eps)

        rms = ( rms / Ngraphene_atoms )**0.5

        # make sure this works, and charges are updated.  If this doesn't work, then we have to reinitialize context which is expensive CPU/GPU communication....
        nbondedForce_Efield.updateParametersInContext(simEfield.context)

        if rms < tol:
            # converged, exit loop
            flag_conv = 1
            break

    # warn if not converged
    if flag_conv == -1:
        print("Warning:  Electrode charges did not converge!!")

    # now update charges
    for i_atom in range(Ngraphene_atoms):
        index = graph[i_atom]  # this is global index of atom for OpenMM data structures
        # for SAPT force field, we don't use sigma, epsilon, if for some reason we start
        # using Lennard-Jones potentials, need to fix code below to update with correct sigma epsilon
        (q_i, sig, eps) = nbondedForce_Efield.getParticleParameters(index)
        nbondedForce.setParticleParameters(index, q_i, 1.0 , 0.0)

    # again make sure this works, and charges are updated
    nbondedForce.updateParametersInContext(simmd.context)


        #**********************  End constant potential code *****************************************************************************


    for j in range(system.getNumForces()):
        f = system.getForce(j)
        print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))






t2 = datetime.now()
t3 = t2 - t1
print('simulation took', t3.seconds,'seconds')
print('Simulating...')
state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True,enforcePeriodicBox=True)
position = state.getPositions()
simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
PDBFile.writeFile(simmd.topology, position, open(strdir+'md_npt.pdb', 'w'))

print('Done!')

exit()
