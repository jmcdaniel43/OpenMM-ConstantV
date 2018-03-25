# OpenMM-ConstantV
This code is for contant potential molecular dynamics simulations within OpenMM.
Electrodes are modeled at the specified potential by solving Gauss's Law with the 
appropriate boundary conditions at each atom of the electrode.  Charges are allowed
to fluctuate with chemical environment to obey the constant potential boundary coundition.

we define two different kinds of graphene sheets.  Some sheets will be neutral (no charge) and some sheets will carry the charge of the electrode.
However, OpenMM doesn't let us define two different residues with the same topology, so we have to use a trick.

for the neutral sheets, we add a dummy hydrogen atom, with no interaction parameters (besides a bond and angle).  These sheets are in the "graph_n" .xml and .pdb files 
