import simtk.unit as unit
import simtk.openmm as mm
import random
import numpy as np

from simtk.openmm.app.dcdreporter import DCDReporter

class VelocityVerletIntegrator(mm.CustomIntegrator):

    """Verlocity Verlet integrator.
    Notes
    -----
    This integrator is taken verbatim from Peter Eastman's example appearing in the CustomIntegrator header file documentation.
    References
    ----------
    W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson, J. Chem. Phys. 76, 637 (1982)
    Examples
    --------
    Create a velocity Verlet integrator.
    >>> timestep = 1.0 * unit.femtoseconds
    >>> integrator = VelocityVerletIntegrator(timestep)
    """

    def __init__(self, timestep=1.0 * unit.femtoseconds):
        """Construct a velocity Verlet integrator.
        Parameters
        ----------
        timestep : np.unit.Quantity compatible with femtoseconds, default: 1*unit.femtoseconds
           The integration timestep.
        """

        super(VelocityVerletIntegrator, self).__init__(timestep)

        self.addPerDofVariable("x1", 0)

        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

class neMDMC(object):

    def __init__(self, simmd, simmc, biasforce, temperature = 298.0 * unit.kelvin, tmc=20.0*unit.picosecond, tboost=5.0*unit.picosecond, nboost=10,numChain=0,numPELen=0,genRandomParticleIdxs=True,biasFirstIdx=-1,biasParticleIdxs=[]):
        self.simmd = simmd
        self.simmc = simmc
        self.stepsize = simmc.integrator.getStepSize()
        self.biasforce = biasforce
        self.tmc = tmc
        self.tboost = tboost
        self.nboost = nboost
        self.booststepsize = int(tboost/(nboost*self.stepsize))
        self.topstepsize = int((tmc-2*tboost)/self.stepsize)
        self.RT = unit.BOLTZMANN_CONSTANT_kB*temperature*unit.AVOGADRO_CONSTANT_NA
        self.naccept = 0
        self.ntrials = 0
        self.numchain = numChain
        self.numpelen = numPELen
        self.useAutoGen = genRandomParticleIdxs
        self.biasidx = biasFirstIdx
        self.biasptclidxs = biasParticleIdxs

    def getBiasFirstIdx(self):
        return self.biasidx

    def setBiasFirstIdx(self,idx):
        self.biasidx = idx

    def setBiasParticleIdxs(self,idxs):
        self.biasptclidxs = idxs

    def setTemperature(self,temperature):
        self.RT = unit.BOLTZMANN_CONSTANT_kB*temperature*unit.AVOGADRO_CONSTANT_NA

    def getTemperature(self):
        return self.RT/(unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA)

    def setTimeMC(self,tmc):
        self.tmc = tmc
        self.topstepsize = int((self.tmc-2*self.tboost)/self.stepsize)

    def getTimeMC(self):
        return self.tmc

    def setTimeMCBoost(self,tboost):
        self.tboost = tboost
        self.booststepsize = int(tboost/(self.nboost*self.stepsize))
        self.topstepsize = int((self.tmc-2*self.tboost)/self.stepsize)

    def getTimeMCBoost(self):
        return self.tboost

    def setNumMCBoost(self,nboost):
        self.nboost = nboost
        self.booststepsize = int(self.tboost/(self.nboost*self.stepsize))

    def getNumMCBoost(self):
        return self.nboost

    def propagate(self):
        self.ntrials += 1
        state = self.simmd.context.getState(getEnergy=True,getVelocities=True,getPositions=True)
        oldE = state.getPotentialEnergy()+state.getKineticEnergy()
        self.simmc.context.setPositions(state.getPositions())
        fflip = random.randint(0,1)*2.-1
        self.simmc.context.setVelocities(fflip*state.getVelocities())
        if(self.useAutoGen):
            if(self.numchain == 0 or self.numpelen == 0):
                print("There are no polymers !")
                return False
            else:
                self.biasidx = self.numpelen * random.randint(0,self.numchain-1)
                self.biasptclidxs = range(self.biasidx,self.biasidx+self.numpelen)
                print('biasidx : '+str(self.biasidx))
        if(len(self.biasptclidxs)==0):
            print("There are no particles to be biased !")
            return False

        fdir = random.randint(0,1)*2.-1

        for i in range(self.nboost):
            if(i==0):
                print(str(fdir))
#                for j in range(len(self.biasptclidxs)):
                    #self.biasforce.setParticleParameters(j,self.biasptclidxs[j],[])
            if(i>0):
#                self.simmc.context.setParameter('lambda',fdir*i/self.nboost)
                for ptcl in self.biasptclidxs:
                    self.biasforce.setParticleParameters(ptcl,ptcl,[fdir*i/self.nboost])
                self.biasforce.updateParametersInContext(self.simmc.context)
#            for j in range(len(self.biasptclidxs)):
#                self.biasforce.setParticleParameters(j,self.biasptclidxs[j],[fdir*i/self.nboost])
            self.simmc.step(self.booststepsize)

#        for j in range(len(self.biasptclidxs)):
#            self.biasforce.setParticleParameters(j,self.biasptclidxs[j],[fdir])
        for ptcl in self.biasptclidxs:
            self.biasforce.setParticleParameters(ptcl,ptcl,[fdir])
        self.biasforce.updateParametersInContext(self.simmc.context)
#        self.simmc.context.setParameter('lambda',fdir)
        self.simmc.step(self.topstepsize)

        for i in range(self.nboost-1,-1,-1):
            for ptcl in self.biasptclidxs:
                self.biasforce.setParticleParameters(ptcl,ptcl,[fdir*i/self.nboost])
#            for j in range(len(self.biasptclidxs)):
#                self.biasforce.setParticleParameters(j,self.biasptclidxs[j],[fdir*i/self.nboost])
            self.biasforce.updateParametersInContext(self.simmc.context)
#            self.simmc.context.setParameter('lambda',fdir*i/self.nboost)
            self.simmc.step(self.booststepsize)

        state = self.simmc.context.getState(getEnergy=True,getVelocities=True,getPositions=True)
        newE = state.getPotentialEnergy()+state.getKineticEnergy()
        if self.metropolis(newE,oldE):
            self.simmd.context.setPositions(state.getPositions())
            self.simmd.context.setVelocities(fflip*state.getVelocities())
            self.naccept += 1
            return True
        else:
            return False

    def getAcceptRatio(self):
        return self.naccept/self.ntrials

    def metropolis(self,pecomp,peref):
        if pecomp < peref:
            return True
        elif (random.uniform(0.0,1.0) < np.exp(-(pecomp - peref)/self.RT)):
            return True
        else:
            return False


class Barostat(object):

    def __init__(self, simeq, pressure = 1.0*unit.bar, temperature = 298.0 * unit.kelvin, barofreq = 25, firstidx=0, secondidx=0, cat_idx=0, shiftscale = 0.2):
        self.shiftscale = shiftscale
        self.simeq = simeq
        self.temperature = temperature
        self.barofreq = barofreq
        self.RT = unit.BOLTZMANN_CONSTANT_kB*temperature*unit.AVOGADRO_CONSTANT_NA
        self.naccept = 0
        self.ntrials = 0
        self.naccept2 = 0
        self.ntrials2 = 0
        self.celldim = self.simeq.topology.getUnitCellDimensions()
        self.lenscale = self.celldim[2]*0.01
        self.pressure = pressure*self.celldim[0]*self.celldim[1]*unit.AVOGADRO_CONSTANT_NA
        self.numres = simeq.topology.getNumResidues()
        #self.numatom = simeq.topology.getNumAtoms()
        self.firstidx = firstidx    # Atoms that are earlier than this index will not move
        self.secondidx = secondidx    # Atoms that are earlier than this index will shift, not scale
        self.cat_idx = cat_idx    # Atoms that are earlier than this index will shift, not scale
        #self.ref_idx_list = ref_idx_list

    def getAcceptRatio(self):
        return self.naccept/self.ntrials

    def metropolis(self,pecomp):
        if pecomp < 0*self.RT:
            return True
        elif (random.uniform(0.0,1.0) < np.exp(-(pecomp)/self.RT)):
            return True
        else:
            return False


#    def shiftsheet(self):
    def shiftsheet(self, merge_ref, veclist,system):
#    def shiftsheet(self, merge_ref):
        self.ntrials2 += 1
        statebak = self.simeq.context.getState(getEnergy=True, getPositions=True)
        oldE = statebak.getPotentialEnergy()

        oldpos = statebak.getPositions()
        oldpos0 = np.asarray(oldpos.value_in_unit(unit.nanometer))
        newpos = np.asarray(oldpos.value_in_unit(unit.nanometer))
        Lz_t0 = oldpos0[self.firstidx:self.secondidx, 2][0] - oldpos0[0:self.firstidx, 2][0]
        disp_t0 = oldpos0[self.secondidx:, 2] - oldpos0[0:self.firstidx, 2][0]
        print(oldpos0[0:self.firstidx, 2][0], oldpos0[self.firstidx+1:self.secondidx, 2][0], oldpos0[self.firstidx:self.secondidx, 2][0], oldpos0[self.secondidx+1:, 2][0],  Lz_t0)

        x_ref_old = []
        y_ref_old = []
        z_ref_old = []
        for ref_i in merge_ref:
            x_ref_old.append(oldpos0[ref_i, 0])
            y_ref_old.append(oldpos0[ref_i, 1])
            z_ref_old.append(oldpos0[ref_i, 2])
            #print("ref_xyz", oldpos0[ref_i])
        #print(x_ref_old, y_ref_old, z_ref_old, type(x_ref_old))
        z_disp_t0 = np.asarray(z_ref_old) - oldpos0[0:self.firstidx, 2][0]
        #print("z_disp:", z_disp_t0)
        
        deltalen = self.shiftscale*(random.uniform(0, 1) * 2 - 1) 
        newpos[self.cat_idx:self.secondidx, 2] += deltalen
        Lz_t1 = newpos[self.firstidx:self.secondidx, 2][0] - oldpos0[0:self.firstidx, 2][0]
        z_disp_t1 = z_disp_t0 * (Lz_t1 / Lz_t0)
        #print("z_disp_new:", z_disp_t1, np.shape(z_disp_t1), type(z_disp_t1))
        z_ref_new = z_disp_t1 + oldpos0[0:self.firstidx, 2][0]
        #print("new_z_position", z_ref_new, np.shape(z_ref_new), type(z_ref_new))
        final_pos_x = []
        final_pos_y = []
        final_pos_z = []
        for res_i in range(len(veclist)):
            res_idx = veclist[res_i]
            x_ref_new_idx = x_ref_old[res_i]
            y_ref_new_idx = y_ref_old[res_i]
            z_ref_new_idx = z_ref_new[res_i]
            #print("old_x:", x_ref_new_idx)
            #print("old_y:", y_ref_new_idx)
            for atom_i in res_idx:
                #print("atome_coord of a residue:", atom_i, atom_i[0], atom_i[1], atom_i[2])
                sum_x = atom_i[0] + x_ref_new_idx
                sum_y = atom_i[1] + y_ref_new_idx
                sum_z = atom_i[2] + z_ref_new_idx
                final_pos_x.append(sum_x)
                final_pos_y.append(sum_y)
                final_pos_z.append(sum_z)
        #print(final_pos_z[0:100])
        #print(newpos[self.secondidx:, 0], type(newpos[self.secondidx:, 0]))
        newpos[self.secondidx:, 0] = np.asarray(final_pos_x)
        newpos[self.secondidx:, 1] = np.asarray(final_pos_y)
        newpos[self.secondidx:, 2] = np.asarray(final_pos_z)
        print(newpos[self.secondidx:] )
        
        self.simeq.context.setPositions(newpos)

        statenew = self.simeq.context.getState(getEnergy=True,getPositions=True)
        newE = statenew.getPotentialEnergy()
        for j in range(system.getNumForces()):
            f = system.getForce(j)
            print(type(f), str(self.simeq.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
        
        #w = newE-oldE 
        w = newE-oldE + self.pressure*(deltalen* unit.nanometer) - len(merge_ref) * self.RT*np.log(Lz_t1/Lz_t0)
        print("dE, exp(-dE/RT) ", w, np.exp(-(w)/self.RT))
        if self.metropolis(w):
            self.naccept2 += 1
        else:
            self.simeq.context.setPositions(oldpos)

        print("Accept ratio for sheetmove", self.naccept2/self.ntrials2)
        if self.ntrials2 >= 10:
            if (self.naccept2 < 0.25*self.ntrials2) :
                self.shiftscale /= 1.1
                self.ntrials2 = 0
                self.naccept2 = 0
            elif self.naccept2 > 0.75*self.ntrials2 :
                self.shiftscale *= 1.1
                self.ntrials2 = 0
                self.naccept2 = 0

    def volumnmove(self,test_reporter):
        self.ntrials += 1
        statebak = self.simeq.context.getState(getEnergy=True, getPositions=True)
        oldE = statebak.getPotentialEnergy()

        oldpos = statebak.getPositions()
        newpos0 = np.asarray(oldpos.value_in_unit(unit.nanometer))
        newpos = np.asarray(oldpos.value_in_unit(unit.nanometer))

        boxvec = statebak.getPeriodicBoxVectors()
        oldboxlen = boxvec[2][2]
        deltalen = self.lenscale*(random.uniform(0, 1) * 2 - 1)
        newboxlen = oldboxlen + deltalen

        newpos[self.firstidx:self.secondidx, 2] += 0.5 * deltalen / unit.nanometer
        newpos[self.secondidx:, 2] *= newboxlen / oldboxlen
        self.simeq.context.setPositions(newpos)

        self.simeq.context.setPeriodicBoxVectors(boxvec[0], boxvec[1], mm.Vec3(0, 0, newboxlen / unit.nanometer) * unit.nanometer)
        statenew = self.simeq.context.getState(getEnergy=True,getPositions=True)
        newE = statenew.getPotentialEnergy()
        w = newE-oldE + self.pressure*deltalen - self.numres * self.RT*np.log(newboxlen/oldboxlen)

        # for i in range(100):
        #     try:
        #         f = self.simeq.system.getForce(i)
        #         print(type(f), str(self.simeq.context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()))
        #         print()
        #     except:
        #         pass

        test_reporter.report(self.simeq, statenew)
        print("Accept ratio for volumnmove", self.getAcceptRatio())

        if self.metropolis(w):
            self.naccept += 1
            # print('newE:', str(newE), 'oldE:', str(oldE))
        else:
            self.simeq.context.setPositions(oldpos)
            self.simeq.context.setPeriodicBoxVectors(boxvec[0], boxvec[1], mm.Vec3(0, 0, oldboxlen / unit.nanometer) * unit.nanometer)

        if self.ntrials >= 10:
            if (self.naccept < 0.25*self.ntrials) :
                self.lenscale /= 1.1
                self.ntrials = 0
                self.naccept = 0
            elif self.naccept > 0.75*self.ntrials :
                self.lenscale = min(self.lenscale*1.1, newboxlen*0.3)
                self.ntrials = 0
                self.naccept = 0


    #def step(self,nstep, merge_ref, veclist, system, res1_list, res2_list, res3_list, position):
    def step(self,nstep, merge_ref, veclist, system, catlist, anlist, solvlist, position):
    #def step(self,nstep, merge_ref):
    #def step(self,nstep):
        test_reporter = DCDReporter("baro_report.dcd", 0, enforcePeriodicBox=True)
        statebak = self.simeq.context.getState(getEnergy=True, getPositions=True)
        test_reporter.report(self.simeq, statebak)


        niter = int(nstep/self.barofreq)
        for i in range(niter):
            self.simeq.step(self.barofreq)
            statebak = self.simeq.context.getState(getEnergy=True, getPositions=True)
            position = statebak.getPositions()
            self.simeq.context.setPositions(position)
            #self.simeq.context.setPositions(newpos)

#            newveclist = []
#            if len(res1_list.get_vectors(position)) == 0:
#                pass
#            else:
#                newveclist.extend(res1_list.get_vectors(position))
#                #print(catlist.get_vectors(position), np.shape(catlist.get_vectors(position)))
#
#            if len(res2_list.get_vectors(position)) == 0:
#                pass
#            else:
#                newveclist.extend(res2_list.get_vectors(position))
#                #print(anlist.get_vectors(position), np.shape(anlist.get_vectors(position)))
#
#            if len(res3_list.get_vectors(position)) == 0:
#                pass
#            else:
#                newveclist.extend(res3_list.get_vectors(position))
#                #print(solvlist.get_vectors(position), np.shape(solvlist.get_vectors(position)))

            newveclist = []
            if len(catlist.get_vectors(position)) == 0:
                pass
            else:
                newveclist.extend(catlist.get_vectors(position))
                #print(catlist.get_vectors(position), np.shape(catlist.get_vectors(position)))

            if len(anlist.get_vectors(position)) == 0:
                pass
            else:
                newveclist.extend(anlist.get_vectors(position))
                #print(anlist.get_vectors(position), np.shape(anlist.get_vectors(position)))

            if len(solvlist.get_vectors(position)) == 0:
                pass
            else:
                newveclist.extend(solvlist.get_vectors(position))
                #print(solvlist.get_vectors(position), np.shape(solvlist.get_vectors(position)))            
            self.shiftsheet(merge_ref, newveclist, system)            
            #self.shiftsheet(merge_ref)            
            #self.shiftsheet()
            #self.volumnmove(test_reporter)

        state = self.simeq.context.getState(getPositions=True, enforcePeriodicBox=True)
        boxVecs = state.getPeriodicBoxVectors()
        return boxVecs


class Graph_list:
    def __init__(self, resname, simeq):
        self.simeq = simeq
        self.resname = resname
        self.res_idx = -1
        self.grpc_1 = -1
        self.grpc_2 = -1
        self.cathode = []
        self.anode = []
        self.steps = []
    def grpclist(self):
        for res in self.simeq.topology.residues():
            if res.name == self.resname:
                if self.res_idx == -1:
                    self.res_idx = res.index
                    for atom in res._atoms:
                        self.cathode.insert(int(atom.name[1:]), atom.index)
                        if atom.name == "C799":
                            self.grpc_1 = atom.index
                elif res.index != self.res_idx:
                    for atom in res._atoms:
                        self.anode.insert(int(atom.name[1:]), atom.index)
                        if atom.name == "C799":
                            self.grpc_2 = atom.index
            if res.name == self.resname:
                for atom in res._atoms:
                    #(q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                    self.steps.append( int(atom.index) )

        return self.grpc_1, self.grpc_2

class all_atomlist:
    def __init__(self, resname, simeq):
        self.simeq = simeq
        self.resname = resname
        self.atomlist = []
        self.Natom = 0
        self.Nres = 0
    def res_list(self):
        for res in self.simeq.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                    self.Natom = int(len(res._atoms))
                    self.atomlist.append( int(atom.index) )
        #self.Natom += int(len(res._atoms))
        self.Nres += int(len(self.atomlist)/self.Natom) if self.Natom != 0 else 0
        #print(self.Nres, self.Natom)
        return self.atomlist, self.Nres, self.Natom


class ref_atomlist:
    def __init__(self, atomlist, Nresidue, Natom_of_1res):
        self.atomlist = atomlist
        self.Nresidue = Nresidue
        self.Natom_of_1res = Natom_of_1res 
        self.atomlist2 = np.reshape(self.atomlist, (self.Nresidue, self.Natom_of_1res)) 
        self.vec = []
        self.ref_list = []
    def get_ref_list(self):
        if len(self.atomlist2) != 0:
            for res_i in range(len(self.atomlist2[:,0])):
                res_idx = self.atomlist2[res_i,0]
                self.ref_list.append(res_idx)
        return np.ravel(self.ref_list)

    def get_vectors(self, position):
        if len(self.atomlist2) == 0:
            pass
        else:
            vec = [[] for i in range(len(self.atomlist2[:,0]))]
            for res_i in range(len(self.atomlist2[:,0])): # residue index
                res_idx = self.atomlist2[res_i,0]
                pos0_res_i = position[res_idx]
                #print("ref1",res_idx, pos0_res_i)
                for atom_i in range(len(self.atomlist2[0,:])): # atoms for each residue
                    atom_idx = self.atomlist2[res_i,atom_i]
                    pos_res_i = position[atom_idx]
                    #print("all_pos, ref_pos",pos_res_i, pos0_res_i)
                    #vec_i = pos_res_i[2]._value - pos0_res_i[2]._value
                    vec_i = pos_res_i._value - pos0_res_i._value
                    #self.vec[res_i].append(np.asarray(vec_i))
                    vec[res_i].append(np.asarray(vec_i))
                    #print(pos_res_i, pos0_res_i, atom_idx )
                    #print(vec_i)
                    #self.allatoms[res_i].append(atom_idx)
            self.vec = vec
        return self.vec
        #return vec


