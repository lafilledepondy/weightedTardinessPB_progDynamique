from subgradients import subgradient_basic, subgradient_Polyak, subgradient_ADS
import numpy as np

def main():
    initial_pi = np.array([3.0, 1.0])
    initial_mu = np.array([])

    # Example on P1
    print(subgradient_Polyak(initial_pi, initial_mu, 0.000001, problem_name="P1")[0])

    # Example on P2 (1 inequality constraint => 1 multiplier in pi)
    print(subgradient_Polyak(np.array([3.0]), np.array([]), 0.000001, problem_name="P2")[0])

if __name__ == "__main__":
    main()