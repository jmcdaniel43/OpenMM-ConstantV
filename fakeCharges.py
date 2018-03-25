#!/usr/bin/env python

"""

Testing script for charge visualization in VMD

"""

from random import random

def makeCharges():
    with open("charges.py.dat", "w") as outfile:
        for i in range(717):
            for j in range(1600):
                if j < 800:
                    rand = random() * 0.5
                else:
                    rand = random() * 0.5 + 0.5
                outfile.write("{:f} ".format(rand))
            outfile.write("\n")

def countCharges():
    with open("charges.dat") as infile:
        charges = []
        chargeMax = 0
        chargeMin = 0
        for line in infile:
            lineCharges = list(map(lambda x: float(x), line.split()[:-1]))
            #print(lineCharges[:10])
            charges.append(lineCharges)
            chargeMax = max(chargeMax, max(lineCharges))
            chargeMin = min(chargeMin, min(lineCharges))
        print(chargeMax, chargeMin)

if __name__ == "__main__":
    countCharges()
