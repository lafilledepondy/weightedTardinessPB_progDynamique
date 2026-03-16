from readData import *
from progDyn import *
from test import *

import numpy as np

def isUnique(sequence):
    return len(np.unique(np.array(sequence))) == len(sequence)

def optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes):
    total_time = sum(processingTimes[i] for i in sequence)
    # Case 1 : borne duale trouver est feasible for original problem alors optimal for intial pb (par th de dualite forte)
    if len(sequence) == nbItems and isUnique(sequence) and total_time == T:
        return "Solution réalisable (donc optimale)"
    # Case 2 : relaxation solution but not feasible for intial pb 
    elif total_time == T:
        return "Solution NON réalisable (borne dual)"
    else:
        return "ERROR"

def main():
    print("")

    datafilePath = 'data/wt040/wt040_001.dat'

    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax1(nbItems, T, processingTimes, dueDates, penalties)
    print(optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes))


    
if __name__ == "__main__":
    main() 