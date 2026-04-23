from subgradients import subgradient_basic, subgradient_Polyak, subgradient_ADS, cutting_planes
from readData_progDyn import *
from progDyn import relax1, relax2, optimalOrRealisableOrInfesable
from test_progDyn import *
from problems import SchedulingInstance
from pathlib import Path


import numpy as np

def demo_progDyn():
    project_root = Path(__file__).resolve().parents[1]
    datafilePath = project_root / 'data_aone' / 'wt040' / 'wt040_001.dat'

    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax1(nbItems, T, processingTimes, dueDates, penalties)
    print(optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes))
    

def main():
    initial_pi = np.array([3.0, 1.0])
    initial_mu = np.array([])

    # print("P1")
    # print("Subgradient Basic:", subgradient_basic(initial_pi, initial_mu, 0.000001, problem_name="P1")[0])
    # print("Subgradient Polyak:", subgradient_Polyak(initial_pi, initial_mu, 0.000001, problem_name="P1")[0])
    # print("Subgradient ADS:", subgradient_ADS(initial_pi, initial_mu, 0.000001, problem_name="P1")[0])
    # print("Cutting Planes:", cutting_planes(0.000001, problem_name="P1")[1])

    # print("P2")
    # print("Subgradient Basic:", subgradient_basic(initial_pi, initial_mu, 0.000001, problem_name="P2")[0])
    # print("Subgradient Polyak:", subgradient_Polyak(initial_pi, initial_mu, 0.000001, problem_name="P2")[0])
    # print("Subgradient ADS:", subgradient_ADS(initial_pi, initial_mu, 0.000001, problem_name="P2")[0])
    # print("Cutting Planes:", cutting_planes(0.000001, problem_name="P2")[1])

    print("P3-Weithed Tardiness Problem")
    project_root = Path(__file__).resolve().parents[1]
    datafilePath = project_root / 'data_aone' / 'wt040' / 'wt040_001.dat'
    instance = SchedulingInstance.from_file(datafilePath)

    initial_pi_p3 = np.zeros(instance.horizon + 1, dtype=float)
    initial_mu_p3 = np.zeros(instance.nb_jobs, dtype=float)

    print("Subgradient Basic:", subgradient_basic(initial_pi_p3, initial_mu_p3, 0.000001, problem_name="P3", instance=instance)[0])
    print("Subgradient Polyak:", subgradient_Polyak(initial_pi_p3, initial_mu_p3, 0.000001, problem_name="P3", instance=instance)[0])
    print("Subgradient ADS:", subgradient_ADS(initial_pi_p3, initial_mu_p3, 0.000001, problem_name="P3", instance=instance)[0])
    # print("Cutting Planes:", cutting_planes(0.000001, problem_name="P3", instance=instance)[1])

    # demo_progDyn()    
    

if __name__ == "__main__":
    main()
   
    
    