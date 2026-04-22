from subgradients import subgradient_basic, subgradient_Polyak, subgradient_ADS
from readData_progDyn import *
from progDyn import relax1, relax2, optimalOrRealisableOrInfesable
from test_progDyn import *

import numpy as np

def demo_progDyn():
    datafilePath = 'data/wt040/wt040_001.dat'

    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax1(nbItems, T, processingTimes, dueDates, penalties)
    print(optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes))
    

def main():
    initial_pi = np.array([3.0, 1.0])
    initial_mu = np.array([])

    # Example on P1 
    print(subgradient_Polyak(initial_pi, initial_mu, 0.000001, problem_name="P1")[0])

    # Example on P2 
    print(subgradient_Polyak(np.array([3.0]), np.array([]), 0.000001, problem_name="P2")[0])

if __name__ == "__main__":
    main()