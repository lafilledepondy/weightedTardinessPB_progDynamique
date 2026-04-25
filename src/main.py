from demos import *
import numpy as np
from pathlib import Path
import time


def _parse_vector_input(raw_text):
    cleaned = raw_text.strip()
    if cleaned == "":
        return np.array([], dtype=float)

    parts = [part.strip() for part in cleaned.split(",")]
    if any(part == "" for part in parts):
        raise ValueError("Empty value in vector input")
    return np.array([float(part) for part in parts], dtype=float)


def pseudoMain():
    project_root = Path(__file__).resolve().parents[1]

    while True:
        algo_choice = input("Type: yes for Prog Dyn ; no for subgradient\n> ").strip().lower()
        if algo_choice in {"yes", "no"}:
            break
        print("Invalid input. Type 'yes' or 'no'.")

    if algo_choice == "yes":
        while True:
            datafile_raw = input("dataFile\n> ").strip()
            datafile_path = Path(datafile_raw)
            if not datafile_path.is_absolute():
                datafile_path = project_root / datafile_path

            if datafile_path.exists() and datafile_path.is_file():
                break
            print("Invalid data file path. Please provide an existing file path.")

        start = time.perf_counter()
        nbItems, T, processingTimes, dueDates, penalties = readData(datafile_path)
        L_tab, sequence = relax1(nbItems, T, processingTimes, dueDates, penalties)
        status = optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes)
        elapsed = time.perf_counter() - start

        print(status)
        print(f"Time taken: {elapsed:.6f} seconds")
        return

    while True:
        program_choice = input("which programme P1, P2, P3, P4, type 1, 2, 3, or 4\n> ").strip()
        if program_choice in {"1", "2", "3", "4"}:
            break
        print("Invalid input. Type 1, 2, 3, or 4.")

    while True:
        sg_choice = input("which subgradient 1 for Basic, 2 for ADS, 3 for Polyak, 4 for Cutting planes\n> ").strip()
        if sg_choice in {"1", "2", "3", "4"}:
            break
        print("Invalid input. Type 1, 2, 3, or 4.")

    problem_name = f"P{program_choice}"
    min_step_size = 0.000001
    max_iterations = 100

    if problem_name in {"P1", "P2"}:
        initial_pi = np.array([3.0, 1.0])
        initial_mu = np.array([])
        instance = None
    else:
        while True:
            datafile_raw = input("dataFile for selected program (P3/P4)\n> ").strip()
            datafile_path = Path(datafile_raw)
            if not datafile_path.is_absolute():
                datafile_path = project_root / datafile_path

            if datafile_path.exists() and datafile_path.is_file():
                break
            print("Invalid data file path. Please provide an existing file path.")

        instance = SchedulingInstance.from_file(datafile_path)
        initial_pi = np.ones(instance.horizon + 1, dtype=float)
        initial_mu = np.ones(instance.nb_jobs, dtype=float)

    while True:
        custom_params_choice = input(
            "Custom subgradient parameters? type yes or no\n> "
        ).strip().lower()
        if custom_params_choice in {"yes", "no"}:
            break
        print("Invalid input. Type 'yes' or 'no'.")

    if custom_params_choice == "yes":
        while True:
            try:
                default_pi_str = ",".join(str(v) for v in initial_pi.tolist())
                pi_raw = input(
                    f"initial_pi as comma-separated floats (default: {default_pi_str})\n> "
                ).strip()
                if pi_raw != "":
                    custom_pi = _parse_vector_input(pi_raw)
                    if problem_name in {"P3", "P4"} and custom_pi.size != initial_pi.size:
                        print(
                            f"For {problem_name}, initial_pi must contain {initial_pi.size} values."
                        )
                        continue
                    initial_pi = custom_pi
                break
            except ValueError:
                print("Invalid vector format. Example: 3, 1")

        while True:
            try:
                default_mu_str = ",".join(str(v) for v in initial_mu.tolist())
                mu_raw = input(
                    f"initial_mu as comma-separated floats (default: {default_mu_str})\n> "
                ).strip()
                if mu_raw != "":
                    custom_mu = _parse_vector_input(mu_raw)
                    if problem_name in {"P3", "P4"} and custom_mu.size != initial_mu.size:
                        print(
                            f"For {problem_name}, initial_mu must contain {initial_mu.size} values."
                        )
                        continue
                    initial_mu = custom_mu
                break
            except ValueError:
                print("Invalid vector format. Example: 1, 1, 1")

        while True:
            step_raw = input("min_step_size (default: 0.000001)\n> ").strip()
            if step_raw == "":
                break
            try:
                candidate = float(step_raw)
                if candidate <= 0:
                    print("min_step_size must be strictly positive.")
                    continue
                min_step_size = candidate
                break
            except ValueError:
                print("Invalid number format. Enter a positive float.")

        while True:
            iter_raw = input("iter_max (default: 100)\n> ").strip()
            if iter_raw == "":
                break
            try:
                candidate = int(iter_raw)
                if candidate <= 0:
                    print("iter_max must be a positive integer.")
                    continue
                max_iterations = candidate
                break
            except ValueError:
                print("Invalid integer format. Enter a positive integer.")

    start = time.perf_counter()

    if sg_choice == "1":
        lb_value, _, _ = subgradient_basic(
            initial_pi,
            initial_mu,
            min_step_size,
            problem_name=problem_name,
            instance=instance,
            max_iterations=max_iterations,
        )
    elif sg_choice == "2":
        lb_value, _, _ = subgradient_ADS(
            initial_pi,
            initial_mu,
            min_step_size,
            problem_name=problem_name,
            instance=instance,
            max_iterations=max_iterations,
        )
    elif sg_choice == "3":
        lb_value, _, _ = subgradient_Polyak(
            initial_pi,
            initial_mu,
            min_step_size,
            problem_name=problem_name,
            instance=instance,
            max_iterations=max_iterations,
        )
    else:
        _, lb_value, _ = cutting_planes(
            min_step_size,
            problem_name=problem_name,
            instance=instance,
            max_iterations=max_iterations,
        )

    elapsed = time.perf_counter() - start
    print(f"LB value: {lb_value}")
    print(f"Time taken: {elapsed:.6f} seconds")


def main():
    # print("\nSOUS-GRADIENTS")
    # demo_p3()  
    # demo_p4()

    # p1_allSG(initial_pi=np.array([3, 1]))
    # p1_allSG(initial_pi=np.array([0, 0]))
    # p1_allSG(initial_pi=np.array([10, 10]))

    # p2_allSG(initial_pi=np.array([3, 1]))
    # p2_allSG(initial_pi=np.array([0, 0]))
    # p2_allSG(initial_pi=np.array([10, 10]))

    # p3_allSG("data_ocsc/wt040/wt040_001.dat")
    # p3_allSG("data_ocsc/wt040/wt040_005.dat")
    # p3_allSG("data_ocsc/wt040/wt040_025.dat")
    # p3_allSG("data_ocsc/wt050/wt050_001.dat")
    # p3_allSG("data_ocsc/wt050/wt050_005.dat")

    # p4_allSG("data_ocsc/wt040/wt040_001.dat")
    # p4_allSG("data_ocsc/wt040/wt040_005.dat")
    # p4_allSG("data_ocsc/wt040/wt040_025.dat")
    # p4_allSG("data_ocsc/wt050/wt050_001.dat")
    # p4_allSG("data_ocsc/wt050/wt050_005.dat")

    # print("\nPROG. DYNAMIQUE")
    # demo_progDyn()  

    pseudoMain()


if __name__ == "__main__":
    main()
   
    
    