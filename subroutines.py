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

#from run_openMM_test_0816 import graph

class MDsimulation:
    def __init__(self, read_pdb, ResidueConnectivityFiles, FF_files, FF_Efield_files):
        strdir = '../'
        pdb = PDBFile( strdir + read_pdb)
        self.temperature = 300*kelvin
        self.cutoff = 1.4*nanometer

        integ_md = DrudeLangevinIntegrator(self.temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
        integ_md.setMaxDrudeDistance(0.02)  # this should prevent polarization catastrophe during equilibration, but shouldn't affect results afterwards ( 0.2 Angstrom displacement is very large for equil. Drudes)
        integ_junk = DrudeLangevinIntegrator(self.temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)  # this integrator won't be used, its just for Efield electric field calculation
        for bond in ResidueConnectivityFiles:
            pdb.topology.loadBondDefinitions(bond) 
            #print(bond)
        pdb.topology.createStandardBonds()

        modeller = Modeller(pdb.topology, pdb.positions)
        forcefield = ForceField(*FF_files)
        modeller.addExtraParticles(forcefield)

        modeller2 = Modeller(pdb.topology, pdb.positions)
        forcefield_Efield = ForceField(*FF_Efield_files)
        modeller2.addExtraParticles(forcefield_Efield)
        
        self.system = forcefield.createSystem(modeller.topology, nonbondedCutoff=self.cutoff, constraints=None, rigidWater=True)
        self.nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        self.customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        self.drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        self.nbondedForce.setNonbondedMethod(NonbondedForce.PME)
        self.customNonbondedForce.setNonbondedMethod(min(self.nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))
        print('nbMethod : ', self.customNonbondedForce.getNonbondedMethod())

        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            type(f)
            f.setForceGroup(i)
            # Here we are adding periodic boundaries to intra-molecular interactions.  Note that DrudeForce does not have this attribute, and
            # so if we want to use thole screening for graphite sheets we might have to implement periodic boundaries for this force type
            if type(f) == HarmonicBondForce or type(f) == HarmonicAngleForce or type(f) == PeriodicTorsionForce or type(f) == RBTorsionForce:
                f.setUsesPeriodicBoundaryConditions(True)
            f.usesPeriodicBoundaryConditions()

        # set up system2 for Efield calculation
        self.system_Efield = forcefield_Efield.createSystem(modeller2.topology, nonbondedCutoff=self.cutoff, constraints=None, rigidWater=True) 
        self.nbondedForce_Efield = [f for f in [self.system_Efield.getForce(i) for i in range(self.system_Efield.getNumForces())] if type(f) == NonbondedForce][0]
        self.nbondedForce_Efield.setNonbondedMethod(NonbondedForce.PME)

        for i in range(self.system_Efield.getNumForces()): 
            f = self.system_Efield.getForce(i)
            type(f)
            f.setForceGroup(i)

        totmass = 0.*dalton
        for i in range(self.system.getNumParticles()):
            totmass += self.system.getParticleMass(i)

        platform = Platform.getPlatformByName('CUDA')
        #platform = Platform.getPlatformByName('OpenCL')
        #properties = {'OpenCLPrecision': 'mixed'}
        self.simmd = Simulation(modeller.topology, self.system, integ_md, platform)
        self.simmd.context.setPositions(modeller.positions)

        # set up simulation for Efield
        self.simEfield = Simulation(modeller2.topology, self.system_Efield, integ_junk, platform)
        
        platform = self.simmd.context.getPlatform()
        platformname = platform.getName();
        print(platformname)

	# Initialize energy
        state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)	
        print('Initial Energy')
        print(str(state.getKineticEnergy()))
        print(str(state.getPotentialEnergy()))
        for j in range(self.system.getNumForces()):
            f = self.system.getForce(j)
            print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

	# write initial pdb with drude oscillators
        position = state.getPositions()
        self.simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        PDBFile.writeFile(self.simmd.topology, position, open('start_drudes.pdb', 'w'))
        
        self.simmd.reporters = []
        self.simmd.reporters.append(DCDReporter('md_nvt.dcd', 1000))
        self.simmd.reporters.append(CheckpointReporter('md_nvt.chk', 10000))
        self.simmd.reporters[1].report(self.simmd,state)

        self.flagexclusions = {}

    def equilibration(self):
	# print('Equilibrating...')
	# for i in range(5000):
	#     self.simmd.step(1000)
        state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True,getParameters=True)
        position = state.getPositions()
        self.simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        # PDBFile.writeFile(self.simmd.topology, position, open(strdir+'equil_nvt.pdb', 'w'))
        state = self.simmd.context.getState(getPositions=True)
        initialPositions = state.getPositions()
        self.simmd.context.reinitialize()
        self.simmd.context.setPositions(initialPositions)

        state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True,getParameters=True)
        print('Equilibrated Energy')
        print(str(state.getKineticEnergy()))
        print(str(state.getPotentialEnergy()))
        for j in range(self.system.getNumForces()):
            f = self.system.getForce(j)
            print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

    def exlusionNonbondedForce(self, graph):
    #******* JGM ****************
    # add exclusions for intra-sheet non-bonded interactions.
    
    # first figure out which exclusions we already have (1-4 atoms and closer).  The code doesn't
    # allow the same exclusion to be added twice
        for i in range(self.customNonbondedForce.getNumExclusions()):
            (particle1, particle2) = self.customNonbondedForce.getExclusionParticles(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            self.flagexclusions[string1]=1
            self.flagexclusions[string2]=1

    # now add exclusions for every atom pair in each sheet if we don't already have them
    #cathode first.
        for i in range(int(len(graph)/2)):
            indexi = graph[i]
            for j in range(i+1,int(len(graph)/2)):
                indexj = graph[j]
                string1=str(indexi)+"_"+str(indexj)
                string2=str(indexj)+"_"+str(indexi)
                if string1 in self.flagexclusions and string2 in self.flagexclusions:
                    continue
                else:
                    self.customNonbondedForce.addExclusion(indexi,indexj)
                    self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                    self.nbondedForce_Efield.addException(indexi,indexj,0,1,0,True)
    #now anode
        for i in range(int(len(graph)/2),len(graph)):
            indexi = graph[i]
            for j in range(i+1,int(len(graph)/2)):
                indexj = graph[j]
                string1=str(indexi)+"_"+str(indexj)
                string2=str(indexj)+"_"+str(indexi)
                if string1 in self.flagexclusions and string2 in self.flagexclusions:
                    continue
                else:
                    self.customNonbondedForce.addExclusion(indexi,indexj)
                    self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                    self.nbondedForce_Efield.addException(indexi,indexj,0,1,0,True)    

    def initializeCharge(self, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, small, cell_dist):
        sum_Qi_cat = 0.
        sum_Qi_an = 0.
        for i_atom in range(Ngraphene_atoms):
            index = graph[i_atom]
            (q_i, sig, eps) = self.nbondedForce_Efield.getParticleParameters(index)
            if i_atom < Ngraphene_atoms / 2:
                q_i = 1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + Voltage/cell_dist) * conv + small
                sum_Qi_cat += q_i
            else:  # anode
                q_i = -1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + + Voltage/cell_dist ) * conv - small
                sum_Qi_an += q_i
            self.nbondedForce_Efield.setParticleParameters(index, q_i, 1.0 , 0.0)
 
        self.nbondedForce_Efield.updateParametersInContext(self.simEfield.context)
        return sum_Qi_cat, sum_Qi_an

    def ConvergedCharge(self, Niter_max, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, q_max):
        rms = 0.0
        flag_conv = -1
        for i_step in range(Niter_max):
            print("charge iteration", i_step)
 
            state2 = self.simEfield.context.getState(getEnergy=True,getForces=True,getPositions=True)
            for j in range(self.system_Efield.getNumForces()):
                    f = self.system_Efield.getForce(j)
                    print(type(f), str(self.simEfield.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
 
            forces = state2.getForces()
            for i_atom in range(Ngraphene_atoms):
                    index = graph[i_atom]
                    (q_i_old, sig, eps) = self.nbondedForce_Efield.getParticleParameters(index)
                    q_i_old = q_i_old
                    E_z = ( forces[index][2]._value / q_i_old._value ) if q_i_old._value != 0 else 0
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
 
                    self.nbondedForce_Efield.setParticleParameters(index, q_i, sig, eps)
                    rms += (q_i - q_i_old._value)**2
 
            self.nbondedForce_Efield.updateParametersInContext(self.simEfield.context)
 
            #rms = (rms/Ngraphene_atoms)**0.5
            #if rms < tol:
            #    flag_conv = i_step
            #    break
    # warn if not converged
        if flag_conv == -1:
            print("Warning:  Electrode charges did not converge!! rms: %f" % (rms))
        else:
            print("Steps to converge: " + str(flag_conv + 1))

    def FinalCharge(self, Ngraphene_atoms, graph, args, i, chargesFile):
        sumq_cathode=0
        sumq_anode=0
        print('Final charges on graphene atoms')
        for i_atom in range(Ngraphene_atoms):
            index = graph[i_atom]
            (q_i, sig, eps) = self.nbondedForce_Efield.getParticleParameters(index)
            self.nbondedForce.setParticleParameters(index, q_i, 1.0 , 0.0)
 
            if i_atom < Ngraphene_atoms / 2:
                # print charge on one representative atom for debugging fluctuations
                if i_atom == 100:
                    print('index, charge, sum',index, q_i , sumq_cathode )
                sumq_cathode += q_i._value
            else:
                # print charge on one representative atom for debugging fluctuations
                #if i_atom == Ngraphene_atoms/2 + 100:
                if i_atom == Ngraphene_atoms/2 + 100:
                    print('index, charge, sum',index, q_i, sumq_anode )
                sumq_anode += q_i._value
 
            # if we are on a 1000 step interval, write charges to file
            # i starts at 0, so at i = 9, 1000 frames will have occured
            #if i % 10 == 0:
            if ( int(args.nstep) == 1 and i % 10 == 0 ):
                chargesFile.write("{:f} ".format(q_i._value))
 
            if ( int(args.nstep) == 10 and i % 1 == 0 ):
                chargesFile.write("{:f} ".format(q_i._value))
 
        # write newline to charge file after charge write
        #if i % 10 == 0:
        if ( int(args.nstep) == 1 and i % 10 == 0 ):
            chargesFile.write("\n")
        elif ( int(args.nstep) == 10 and i % 1 == 0 ):
            chargesFile.write("\n")
        
        return sumq_cathode, sumq_anode
   
    def Scale_charge(self, Ngraphene_atoms, graph, ana_Q_Cat, ana_Q_An, sumq_cathode, sumq_anode):
    #def Scale_charge(self, Ngraphene_atoms, graph, Q_Cat_ind, Q_An_ind, sumq_cathode, sumq_anode, sum_Qi_cat, sum_Qi_an):
        Q_cat_scale = 0.
        Q_an_scale = 0.
        for i_atom in range(Ngraphene_atoms):
            index = graph[i_atom]
            (q_i_num, sig, eps) = self.nbondedForce_Efield.getParticleParameters(index)
            if i_atom < Ngraphene_atoms / 2:
                q_i = q_i_num * (ana_Q_Cat/ sumq_cathode)
                #q_i = q_i_num * ((Q_Cat_ind + sum_Qi_cat)/ sumq_cathode)
                Q_cat_scale += q_i._value
            else:  # anode
                q_i = q_i_num * (ana_Q_An/ sumq_anode)
                #q_i = q_i_num * ((Q_An_ind + sum_Qi_an)/ sumq_anode)
                Q_an_scale += q_i._value
            self.nbondedForce_Efield.setParticleParameters(index, q_i, sig, eps)
            self.nbondedForce.setParticleParameters(index, q_i, sig, eps)
        self.nbondedForce_Efield.updateParametersInContext(self.simEfield.context)
        print( 'Updated charge on cathode, anode:', Q_cat_scale, Q_an_scale )
        return Q_cat_scale, Q_an_scale


    def PrintFinalEnergies(self):
        self.nbondedForce.updateParametersInContext(self.simmd.context)
 
        state = self.simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
        for j in range(self.system.getNumForces()):
            f = self.system.getForce(j)
            print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))


class Graph_list:
    def __init__(self, resname):
        self.resname = resname
        self.res_idx = -1
        self.c562_1 = -1
        self.c562_2 = -1
        self.cathode = []
        self.anode = []
        self.dummy = []
    def grpclist(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                    (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(atom.index)
                    if q_i._value != 0:
                        print(atom, q_i)
                if self.res_idx == -1:
                    self.res_idx = res.index
                    for atom in res._atoms:
                        self.cathode.insert(int(atom.name[1:]), atom.index)
                        if atom.name == "C562":
                            self.c562_1 = atom.index
                elif res.index != self.res_idx:
                    for atom in res._atoms:
                        self.anode.insert(int(atom.name[1:]), atom.index)
                        if atom.name == "C562":
                            self.c562_2 = atom.index

            if res.name == self.resname:
                for atom in res._atoms:
                    (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                    self.dummy.append( int(atom.index) )


class solution_Hlist:
    def __init__(self, resname):
        self.resname = resname
        self.cation = []
        self.anion = []
        self.solvent = []
        self.He = []
    def cation_hlist(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                    if 'H' in list(atom.name):
                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                        self.cation.append( int(atom.index) )
    def anion_hlist(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                    if 'B' in list(atom.name):
                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                        self.anion.append( int(atom.index) )
    def solvent_hlist(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                    if 'H' in list(atom.name):
                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                        self.solvent.append( int(atom.index) )
    def vac_list(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                        self.He.append( int(atom.index) )


class solution_allatom:
    def __init__(self, resname):
        self.resname = resname
        self.atomlist = []
    def res_list(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                        self.atomlist.append( int(atom.index) )



class get_Efield:
    def __init__(self, alist):
        self.alist = alist
        self.efieldx = []
        self.efieldy = []
        self.efieldz = []
        self.position_z = []
        self.Q_Cat_ind = 0.
        self.Q_An_ind = 0.
        self.Q_Cat = 0.
        self.Q_An = 0.
    def efield(self, sim, forces):
        for H_i in range(len(self.alist)):
            H_idx = self.alist[H_i]
            (q_i, sig, eps) = sim.nbondedForce_Efield.getParticleParameters(H_idx)
            E_x_i = ( forces[H_idx][0]._value / q_i._value ) if q_i._value != 0 else 0
            E_y_i = ( forces[H_idx][1]._value / q_i._value ) if q_i._value != 0 else 0
            E_z_i = ( forces[H_idx][2]._value / q_i._value ) if q_i._value != 0 else 0
            self.efieldx.append( E_x_i )
            self.efieldy.append( E_y_i )
            self.efieldz.append( E_z_i )
    def Pos_z(self, positions):
        for H_i in range(len(self.alist)):
            H_idx = self.alist[H_i]
            self.position_z.append( positions[H_idx][2]._value )
    #def induced_q(self, eletrode_L, eletrode_R, cell_dist, sim, positions):
    def induced_q(self, eletrode_L, eletrode_R, cell_dist, sim, positions, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv):
        for H_i in range(len(self.alist)):
            H_idx = self.alist[H_i]
            (q_i, sig, eps) = sim.nbondedForce_Efield.getParticleParameters(H_idx)
            self.position_z.append( positions[H_idx][2]._value )
            zR = eletrode_R - positions[H_idx][2]._value
            zL = positions[H_idx][2]._value - eletrode_L
            self.Q_Cat_ind += (zR / cell_dist)* (- q_i._value)
            self.Q_An_ind += (zL / cell_dist)* (- q_i._value)
#        return self.Q_Cat_ind, self.Q_An_ind

        for i_atom in range(Ngraphene_atoms):
            index = graph[i_atom]
            (q_i, sig, eps) = sim.nbondedForce_Efield.getParticleParameters(index)
            if i_atom < Ngraphene_atoms / 2:
                q_i = 1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + Voltage/cell_dist) * conv
                self.Q_Cat += q_i
            else:  # anode
                q_i = -1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + Voltage/cell_dist) * conv
                self.Q_An += q_i
        ana_Q_Cat =  self.Q_Cat_ind + self.Q_Cat
        ana_Q_An = self.Q_An_ind + self.Q_An
        return ana_Q_Cat, ana_Q_An


def Distance(p1, p2, initialPositions):
    pos_c562_1 = initialPositions[p1]
    pos_c562_2 = initialPositions[p2]
    cell_dist = 0
    for i in range(3):
        d = pos_c562_1[i] / nanometer - pos_c562_2[i] / nanometer
        cell_dist += (d**2)

    cell_dist = cell_dist**(1/2)
    return(cell_dist, pos_c562_1[2]/nanometer, pos_c562_2[2]/nanometer)


class hist_Efield:
    def __init__(self, dz, zdim, zlist, Ezlist):
        self.dz = dz
        self.zdim = zdim
        self.bins = [i*self.dz for i in range(0, int(self.zdim/self.dz))]
        self.zlist = zlist
        self.Ezlist = Ezlist
        self.Ezcount_i = []
    def Efield(self):
        for bin_i in range(len(self.bins)-1):
            bin0 = self.bins[bin_i]
            bin1 = self.bins[bin_i+1]
            ztotal = [self.Ezlist[i] for i,x in enumerate(self.zlist) if bin0 < x <= bin1]
            avg_count = sum(ztotal)/len(ztotal) if len(ztotal) != 0 else 0
            self.Ezcount_i.append( avg_count )
        return self.Ezcount_i
    def save_hist(self, hlist, filename):
        ofile = open(filename, "w")
        for i in range(len(hlist)):
            line = str("{0:3.5f}".format(float(self.bins[i+1]))) + "  " + str("{0:5.8f}".format(float(hlist[i]))) + "\n"
            ofile.write(line)
        ofile.close()


