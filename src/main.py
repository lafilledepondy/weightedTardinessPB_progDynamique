from subgradients import subgradient_basic, subgradient_Polyak, subgradient_ADS, cutting_planes
from readData_progDyn import *
from progDyn import relax1, relax2, optimalOrRealisableOrInfesable
from test_progDyn import *
from problems import SchedulingInstance, p3_compute_dual_function, p3_compute_ineq_ctrs_functions, p3_compute_objective_function, p3_math_prog_dims, p4_compute_objective_function
from pathlib import Path
import time


import numpy as np

def demo_progDyn():
    project_root = Path(__file__).resolve().parents[1]
    datafilePath = project_root / 'data_aone' / 'wt040' / 'wt040_001.dat'

    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax1(nbItems, T, processingTimes, dueDates, penalties)
    print(optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes))

def demo_p3():
    from problems import SchedulingInstance, p3_compute_dual_function, p3_feasible_x_sol, p3_math_prog_dims
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data_aone' / 'wt040' / 'wt040_051.dat'

    instance = SchedulingInstance.from_file(data_path)

    nb_vars, nb_ineq, nb_eq = p3_math_prog_dims(instance)
    
    print(f"Nombre de jobs : {instance.nb_jobs}")
    print(f"Nombre total de variables (x_it) : {nb_vars}")
    print(f"Nombre de contraintes d'inégalité (T) : {nb_ineq}")

    mu = np.ones(nb_ineq) * 500
    pi = 0

    I= list(range(instance.nb_jobs))
    Ti = {i: list(range(instance.horizon - instance.processing_times[i] + 1)) for i in I}
  
    dual_val, x_opt = p3_compute_dual_function(instance, Ti, pi, mu)
    
    print(f"Valeur dual (LB) : {dual_val}")

    x_feasible = p3_feasible_x_sol(instance, Ti)
    obj_val = p3_compute_objective_function(instance, Ti, x_feasible)
    
    print(f"Opt réalisable (UB) : {obj_val}")

def demo_p4():
    from problems import SchedulingInstance, p4_compute_dual_function, p4_feasible_x_sol, p4_math_prog_dims
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data_aone' / 'wt040' / 'wt040_001.dat'

    instance = SchedulingInstance.from_file(data_path)

    nb_vars, nb_ineq, nb_eq = p4_math_prog_dims(instance)
    
    print(f"Nombre de jobs : {instance.nb_jobs}")
    print(f"Nombre total de variables (x_it) : {nb_vars}")
    print(f"Nombre de contraintes d'inégalité (T) : {nb_ineq}")

    mu = np.ones(nb_ineq) * 500
    pi = 0

    I= list(range(instance.nb_jobs))
    Ti = {i: list(range(instance.horizon - instance.processing_times[i] + 1)) for i in I}
  
    dual_val, x_opt = p4_compute_dual_function(instance, Ti, pi, mu)
    
    print(f"Valeur dual (LB) : {dual_val}")

    x_feasible = p4_feasible_x_sol(instance, Ti)
    obj_val = p4_compute_objective_function(instance, Ti, x_feasible)
    
    print(f"Opt réalisable (UB) : {obj_val}")

def p3_allSG():
    print("P3-Weithed Tardiness Problem")
    project_root = Path(__file__).resolve().parents[1]
    datafilePath = project_root / 'data_ocsc' / 'wt040' / 'wt040_005.dat'
    instance = SchedulingInstance.from_file(datafilePath)

    print("Instance :", datafilePath)

    initial_pi_p3 = np.ones(instance.horizon + 1, dtype=float)
    initial_mu_p3 = np.ones(instance.nb_jobs, dtype=float)

    I= list(range(instance.nb_jobs))
    Ti = {i: list(range(instance.horizon - instance.processing_times[i] + 1)) for i in I}
    start = time.time()
    dual_value_p3, best_x, _ = subgradient_basic(initial_pi_p3, initial_mu_p3, 0.000001, problem_name="P3", instance=instance, max_iterations=100)
    end = time.time()
    print("Subgradient Basic:", dual_value_p3, "Best x:", best_x) # DualValue, Best x
    print("Time taken:", end - start)
    print("____________________________________________________________")
    start = time.time()
    dual_value_p3_polyak, best_x_polyak, _ = subgradient_Polyak(initial_pi_p3, initial_mu_p3, 0.000001, problem_name="P3", instance=instance, max_iterations=100)
    print("Subgradient Polyak:", dual_value_p3_polyak, "Best x:", best_x_polyak) # DualValue
    end = time.time()
    print("Time taken:", end - start)
    print("____________________________________________________________")
    start = time.time()
    dual_value_p3_ads, best_x_p3_ads, _ = subgradient_ADS(initial_pi_p3, initial_mu_p3, 0.000001, problem_name="P3", instance=instance, max_iterations=100)
    print("Subgradient ADS:", dual_value_p3_ads, "Best x:", best_x_p3_ads) # DualValue
    end = time.time()
    print("Time taken:", end - start)
    print("____________________________________________________________")
    start = time.time()
    best_lambda_p3_cutting, dual_value_p3_cutting, ub_p3_cutting = cutting_planes(0.000001, problem_name="P3", instance=instance, max_iterations=100,)
    print("Cutting Planes - LB:", dual_value_p3_cutting, "UB:", ub_p3_cutting, "Best lambda:", best_lambda_p3_cutting)
    end = time.time()
    print("Time taken:", end - start)

def p4_allSG():
    print("P4-Weithed Tardiness Problem")
    project_root = Path(__file__).resolve().parents[1]
    datafilePath = project_root / 'data_ocsc' / 'wt040' / 'wt040_001.dat'
    instance = SchedulingInstance.from_file(datafilePath)

    print("Instance :", datafilePath)

    initial_pi_p3 = np.ones(instance.horizon + 1, dtype=float)
    initial_mu_p3 = np.ones(instance.nb_jobs, dtype=float)

    I= list(range(instance.nb_jobs))
    Ti = {i: list(range(instance.horizon - instance.processing_times[i] + 1)) for i in I}
    # start = time.time()
    # dual_value_p3, best_x, _ = subgradient_basic(initial_pi_p3, initial_mu_p3, 0.000001, problem_name="P3", instance=instance, max_iterations=100)
    # end = time.time()
    # print("Subgradient Basic:", dual_value_p3, "Best x:", best_x) # DualValue, Best x
    # print("Time taken:", end - start)
    # print("____________________________________________________________")
    # start = time.time()
    # dual_value_p3_polyak, best_x_polyak, _ = subgradient_Polyak(initial_pi_p3, initial_mu_p3, 0.000001, problem_name="P3", instance=instance, max_iterations=100)
    # print("Subgradient Polyak:", dual_value_p3_polyak, "Best x:", best_x_polyak) # DualValue
    # end = time.time()
    # print("Time taken:", end - start)
    # print("____________________________________________________________")
    # start = time.time()
    # dual_value_p3_ads, best_x_p3_ads, _ = subgradient_ADS(initial_pi_p3, initial_mu_p3, 0.000001, problem_name="P3", instance=instance, max_iterations=100)
    # print("Subgradient ADS:", dual_value_p3_ads, "Best x:", best_x_p3_ads) # DualValue
    # end = time.time()
    # print("Time taken:", end - start)
    # print("____________________________________________________________")
    start = time.time()
    best_lambda_p3_cutting, dual_value_p3_cutting, ub_p3_cutting = cutting_planes(0.000001, problem_name="P3", instance=instance, max_iterations=100,)
    print("Cutting Planes - LB:", dual_value_p3_cutting, "UB:", ub_p3_cutting, "Best lambda:", best_lambda_p3_cutting)
    end = time.time()
    print("Time taken:", end - start)       

def main():
    print("")
    # initial_pi = np.array([3.0, 1.0])
    # initial_mu = np.array([])

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


    # p3_allSG()

    p4_allSG()

    # demo_progDyn()  
    # demo_p3()  
    # demo_p4()



if __name__ == "__main__":
    main()
   
    
    