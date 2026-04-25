import math
import numpy as np
import numpy.typing as npt # searched on internet for type specification of numpy array[float] to avoid confusion with int or even list
from numpy import linalg as LA
from scipy.optimize import linprog
from problems import (
    SchedulingInstance,
    set_active_problem,
    compute_ineq_ctrs_functions,
    compute_eq_ctrs_functions,
    compute_dual_function,
    compute_objective_function,
    feasible_x_sol,
)

##############################################
#  Code used by several methods              #
# To implement the Lagrangian relaxations of #
# the weighted tardiness scheduling problem, #
# it is recommended to modify the prototypes #
# of the functions to include an "instance"  #
# object to the parameters.                  #
##############################################
# Returns the subgradient split in two parts:
# first, the indices associated with inequality constraints,
# second, the indices associated with equality constraints,
def compute_subgradient(x):
    return compute_ineq_ctrs_functions(x), compute_eq_ctrs_functions(x)

# Projects the set of multipliers associated with inequality constraints
# onto the non-negative orthant (returns the vector of multipliers with 
# negative components set to zero).
def project_solution(pi):
    # np array much faster than list comprehension
    if isinstance(pi, np.ndarray):
        return np.maximum(pi, 0)
    
    # compherension list slow
    return [max(x, 0) for x in pi]

# Returns the new step size
def update_step_size(step_size, alpha=0.8):
    # choix arbitraire 0.8 ; should be decreasing and not too fast
    return alpha * step_size


##############################################
#          Basic subgradient procedure       #
##############################################
def subgradient_basic(
    initial_pi: npt.NDArray[np.float64],
    initial_mu: npt.NDArray[np.float64],
    min_step_size: float,
    problem_name: str,
    instance: SchedulingInstance | None = None,
    initial_step_size: float = 2.0,
    alpha: float = 0.8,
    max_iterations: int | None = None,
):
    set_active_problem(problem_name, instance=instance)
    pi = initial_pi
    mu = initial_mu 

    step_size = float(initial_step_size)

    best_Dualvalue = -math.inf
    best_x = None

    history = [] # to track

    iteration = 0
    while step_size > min_step_size and (max_iterations is None or iteration < max_iterations):
        # solve Lagrangian subproblem
        dual_value, x = compute_dual_function(pi, mu)

        # keep best
        if dual_value > best_Dualvalue:
            best_Dualvalue = dual_value
            best_x = x

        # subgradient
        sg_pi, sg_mu = compute_subgradient(x)

        # update multipliers
        pi = pi + step_size * sg_pi 
        mu = mu + step_size * sg_mu

        # projection pi >= 0
        pi = project_solution(pi) # returns compherension list

        # update step
        step_size = update_step_size(step_size, alpha)

        # update the history
        history.append({
            "dual_value": dual_value,
            "pi": np.copy(pi),
            "mu": np.copy(mu)
        })
        iteration += 1
    return round(best_Dualvalue, 2), best_x, history

##############################################################
#          Subgradient with Polyak step size procedure       #
##############################################################
def update_polyak_step_size(beta_k, L_star, L_k, d_k):
    """
    Polyak's ruke s_k = beta_k * (L_star - L_k) / ||d_k||**2
    L_star: primal bound
    L_k: dual value at iteration k
    d_k: subgradient at iteration k
    beta_k: parameter in (0,1)
    s_k: step size at iteration k
    """
    norm_squared = LA.norm(d_k) ** 2
    
    if norm_squared == 0:
        return 0.0 # */0 = 0 to avoid complication 
    
    return beta_k * (L_star - L_k) / norm_squared

def subgradient_Polyak(
    initial_pi: npt.NDArray[np.float64],
    initial_mu: npt.NDArray[np.float64],
    min_step_size: float,
    problem_name: str,
    instance: SchedulingInstance | None = None,
    beta0: float = 1.0,
    alpha: float = 0.8,
    initial_step_size: float = 2.0,
    max_iterations: int | None = None,
):
    set_active_problem(problem_name, instance=instance)
    pi = initial_pi
    mu = initial_mu 

    step_size = float(initial_step_size)
    beta_k = float(beta0)
    # if beta_k is constant then the method is not guaranteed to converge TAKES to LONG
    # beta_k will be updated down

    L_star = compute_objective_function(feasible_x_sol()) # primal bound

    best_Dualvalue = -math.inf
    best_x = None

    history = [] # to track

    iteration = 0
    while step_size > min_step_size and (max_iterations is None or iteration < max_iterations):
        # solve Lagrangian subproblem
        dual_value, x = compute_dual_function(pi, mu)

        # keep best
        if dual_value > best_Dualvalue:
            best_Dualvalue = dual_value
            best_x = x

        # subgradient
        sg_pi, sg_mu = compute_subgradient(x)

        # update multipliers
        pi = pi + step_size * sg_pi 
        mu = mu + step_size * sg_mu

        # projection pi >= 0
        pi = np.array(project_solution(pi))

        # update step (basic method)
        d_k = np.concatenate([sg_pi, sg_mu])
        step_size = update_polyak_step_size(beta_k, L_star, dual_value, d_k)

        # update beta_k (Polyak's rule)
        beta_k = update_step_size(beta_k, alpha)
        
        # update the history
        history.append({
            "dual_value": dual_value,
            "pi": np.copy(pi),
            "mu": np.copy(mu)
        })
        iteration += 1
    return round(best_Dualvalue, 2), best_x, history


##############################################
#     Deflected subgradient procedure: ADS   #
##############################################
def compute_direction_ADS(sg_pi, sg_mu, direction_pi, direction_mu):
    """
    ADS = Average Direction Strategy
        = d^k = g^k + psi * d^(k-1)
    where: psi = ||g^k|| / ||d^(k-1)||

    if 
        - first iteration => no previous direction (case 1)
        - previous direction = 0 norm  (case 2)
    then d^k = g^k
    """    

    # pi update
    norm_prev_pi = np.linalg.norm(direction_pi)
    norm_sg_pi = np.linalg.norm(sg_pi)

    if norm_prev_pi == 0:
        new_direction_pi = np.copy(sg_pi) # case 1 & case 2 ; case 1 => bcse since direction_pi is initialized to 0 so the norm will become 0
    else:
        psi_pi = norm_sg_pi / norm_prev_pi
        new_direction_pi = sg_pi + psi_pi * direction_pi

        # avoid null direction 
        if np.linalg.norm(new_direction_pi) == 0:
            new_direction_pi = np.copy(sg_pi)

    # mu update 
    # (same logic as pi update) 
    norm_prev_mu = np.linalg.norm(direction_mu)
    norm_sg_mu = np.linalg.norm(sg_mu)

    if norm_prev_mu == 0:
        new_direction_mu = np.copy(sg_mu)
    else:
        psi_mu = norm_sg_mu / norm_prev_mu
        new_direction_mu = sg_mu + psi_mu * direction_mu

        if np.linalg.norm(new_direction_mu) == 0:
            new_direction_mu = np.copy(sg_mu)

    return new_direction_pi, new_direction_mu


def subgradient_ADS(
    initial_pi: npt.NDArray[np.float64],
    initial_mu: npt.NDArray[np.float64],
    min_step_size: float,
    problem_name: str,
    instance: SchedulingInstance | None = None,
    max_iterations: int | None = None
):
    set_active_problem(problem_name, instance=instance)
    pi = initial_pi
    mu = initial_mu 

    # previous directions 
    direction_pi = np.zeros_like(pi) # intialized to 0 it's why in the compute_direction_ADS for the case 1 we permit to use norm=0
    direction_mu = np.zeros_like(mu)

    step_size = 2.0

    best_Dualvalue = -math.inf
    best_x = None

    history = []

    iteration = 0
    while step_size > min_step_size and (max_iterations is None or iteration < max_iterations):
        # solve Lagrangian subproblem
        dual_value, x = compute_dual_function(pi, mu)

        # keep best
        if dual_value > best_Dualvalue:
            best_Dualvalue = dual_value
            best_x = x

        # subgradient
        sg_pi, sg_mu = compute_subgradient(x)

        # ADS direction
        direction_pi, direction_mu = compute_direction_ADS(
            sg_pi, sg_mu, direction_pi, direction_mu
        )

        # update multipliers
        pi = pi + step_size * sg_pi 
        mu = mu + step_size * sg_mu

        # projection pi >= 0
        pi = np.array(project_solution(pi), dtype=float) # returns compherension list then converted to npArray

        # update step
        step_size = update_step_size(step_size)

        # update the history
        # we save all for reuse
        history.append({
            "dual_value": dual_value,
            "best_dual": best_Dualvalue,
            "step": step_size,
            "pi": np.copy(pi),
            "mu": np.copy(mu)
        })
        iteration += 1
        
    return round(best_Dualvalue, 2), best_x, history

##############################################
#                Cutting planes              #
##############################################
def initialize_master_program(nb_ineq_ctrs, nb_eq_ctrs, initial_x_sol):
    """
    (D^1_Epi) ; Algorithm 3.4.1
    max t
    s.t. t <= f(x^k) + lambda^T g(x^k)     for generated cuts
        lambda >= 0   

    to initialize we are generating the 1st cut with the initial solution x1 (feasible) 
    """

    # first cut:
    # t <= f(x1) + lambda^T g(x1)
    f_x1 = compute_objective_function(initial_x_sol)
    g_x1 = compute_ineq_ctrs_functions(initial_x_sol)
    h_x1 = compute_eq_ctrs_functions(initial_x_sol)

    master_data = {
        "nb_lambda_ineq": nb_ineq_ctrs,
        "nb_lambda_eq": nb_eq_ctrs,
        "cuts": []
    }

    master_data["cuts"].append({
        "f_val": f_x1,
        "g_val": np.array(g_x1, dtype=float),
        "h_val": np.array(h_x1, dtype=float)
    })

    return master_data


def solve_master_program(master_data):
    """
    solving the master program (D^1_Epi) with the cuts generated so far
    """
    m_ineq = master_data["nb_lambda_ineq"]
    m_eq = master_data["nb_lambda_eq"]
    m = m_ineq + m_eq
    cuts = master_data["cuts"]

    # maximize t <=> minimize -t
    c = np.zeros(m + 1)
    c[-1] = -1.0

    A_ub = []
    b_ub = []

    for cut in cuts:
        row = np.zeros(m + 1)

        # t <= f(x^k) + pi^T g(x^k) + mu^T h(x^k)
        row[:m_ineq] = -cut["g_val"]
        row[m_ineq:m] = -cut["h_val"]
        row[-1] = 1.0

        A_ub.append(row)
        b_ub.append(cut["f_val"])

    bounds = [(0, None)] * m_ineq + [(None, None)] * m_eq + [(None, None)]

    res = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        raise RuntimeError("Master program not solved.")

    pi_r = np.array(res.x[:m_ineq])
    mu_r = np.array(res.x[m_ineq:m])
    t_r = res.x[-1]

    return pi_r, mu_r, t_r


def cutting_planes(
    epsilon,
    problem_name: str,
    instance: SchedulingInstance | None = None,
    return_history: bool = False,
    max_iterations: int | None = None,
):
    set_active_problem(problem_name, instance=instance)
    x1 = feasible_x_sol()

    nb_ineq_ctrs = len(compute_ineq_ctrs_functions(x1))
    nb_eq_ctrs = len(compute_eq_ctrs_functions(x1))

    master_data = initialize_master_program(
        nb_ineq_ctrs,
        nb_eq_ctrs,
        x1
    )

    r = 2
    UB = math.inf
    LB = -math.inf

    best_lambda = np.zeros(nb_ineq_ctrs + nb_eq_ctrs)
    history = []
    iteration = 0

    # main loop Algorithm 3.4.1
    while UB - LB > epsilon and (max_iterations is None or iteration < max_iterations):
        pi_r, mu_r, t_r = solve_master_program(master_data)
        L_lambda_r, x_bar_r = compute_dual_function(pi_r, mu_r)
        UB = t_r

        if L_lambda_r > LB:
            LB = L_lambda_r
            best_lambda = np.concatenate([pi_r, mu_r]).copy()

        if UB - LB > epsilon:
            new_cut = {
                "f_val": compute_objective_function(x_bar_r),
                "g_val": np.array(
                    compute_ineq_ctrs_functions(x_bar_r),
                    dtype=float
                ),
                "h_val": np.array(
                    compute_eq_ctrs_functions(x_bar_r),
                    dtype=float
                )
            }

            master_data["cuts"].append(new_cut)
            r += 1

        history.append({
            "dual_value": float(L_lambda_r),
            "lower_bound": float(LB),
            "upper_bound": float(UB),
            "gap": float(UB - LB),
        })
        iteration += 1

    if return_history:
        return best_lambda, round(LB, 2), round(UB, 2), history

    return best_lambda, round(LB, 2), round(UB, 2)       
