import numpy as np
import numpy
import math
from numpy import linalg as LA

# #########################################################################
# Uncomment the part of code that defines the problem you want to solve. #
# Note: an object-oriented programming approach seems more appropriate,  #
# but might slightly more complicated to implement. Feel free to choose  #
# your preferred approach.                                               #
# In the following, the dual variables associated with <= constraints    #
# are named pi.                                                          #   
# To implement the Lagrangian relaxations of the weighted tardiness      #
# scheduling problem, it is recommended to modify the prototypes of the  #
# functions to include an "instance" object to the parameters.           #
# #########################################################################


# ##############################################
# # Functions defining mathematical program P1 #
# ##############################################

# number of elements of the mathematical program
# returns nb_variables, nb_ineq_ctrs, nb_eq_ctrs
def math_prog_dims():
    return 2,2,0 # 2 varibales, 2 fct g and 0 fct h

# returns the value of the objective function at x
def compute_objective_function(x):
    return x[0]-4*x[1]

# returns the value of the functions defining the inequality constraints at x
def compute_ineq_ctrs_functions(x):
    return numpy.array([ -x[0]-x[1] + 2, x[1] -1  ])

# returns the value of the functions defining the equality constraints at x
def compute_eq_ctrs_functions(x):
    return numpy.array([])

# compute the value of the dual function for given multipliers 
# pi (for inequality constraints) and mu (for equality constraints)
# returns the value, and an optimal solution of the Lagrange subproblem
def compute_dual_function(pi,mu):
    lagrange_costs = [1-pi[0],-4-pi[0]+pi[1]]
    optimal_sol = [(3 if v<0 else 0) for v in lagrange_costs]
    return numpy.dot(optimal_sol,lagrange_costs) + numpy.dot([2,-1],pi), optimal_sol

# returns an arbitrary feasible solution
def feasible_x_sol():
    return [3,1]
##################################
# End of mathematical program P1 #
##################################

# ##############################################
# # Functions defining mathematical program P2 #
# ##############################################

# number of elements of the mathematical program
# returns nb_variables, nb_ineq_ctrs, nb_eq_ctrs
# def math_prog_dims():
#     return 2,1,0

# # returns the value of the objective function at x
# def compute_objective_function(x):
#     return (x[0]-2)*(x[0]-2)+0.25*x[1]*x[1]

# # returns the value of the functions defining the inequality constraints at x
# def compute_ineq_ctrs_functions(x):
#     return numpy.array([ x[0]-3.5*x[1]-1 ])

# # returns the value of the functions defining the equality constraints at x
# def compute_eq_ctrs_functions(x):
#     return numpy.array([])

# # compute the value of the dual function for given multipliers 
# # pi (for inequality constraints) and mu (for equality constraints)
# # returns the value, and an optimal solution of the Lagrange subproblem
# def compute_dual_function(pi,mu):
#     optimal_sol=[2-1.5*pi[0],pi[0]]
#     return -2.5*pi[0]*pi[0]+pi[0], optimal_sol

# # returns an arbitrary feasible solution
# def feasible_x_sol():
#     return [0,4/3]
##################################
# End of mathematical program P2 #
##################################

###################################################################################
# Functions required for the weighted tardiness single machine scheduling problem #
###################################################################################
class SchedulingInstance:
    """
    Represents a single-machine scheduling instance with weighted tardiness data.
    Attributes:
        nb_jobs (int): Number of jobs in the instance.
        processing_times (list[int]): Processing time for each job.
        weights (list[int]): Penalty weight for each job.
        due_dates (list[int]): Due date for each job.
        horizon (int): Total processing time of all jobs (planning horizon).
    Methods:
        from_file(file_path):
            Build and return a SchedulingInstance from a text file.
            Expected file format:
                - First non-empty line: number of jobs (integer).
                - Next non-empty lines: one job per line with
                  "processing_time due_date weight".
            Raises:
                ValueError: If the file is empty, malformed, or job count is inconsistent.
    """
    def __init__(self, nb_jobs, processing_times, weights, due_dates):
        self.nb_jobs=nb_jobs
        self.processing_times=processing_times
        self.weights=weights
        self.due_dates=due_dates
        self.horizon=sum(processing_times)
    
    # read from file
    @staticmethod
    def from_file(file_path):
        with open(file_path) as f:
            first_line = f.readline().strip()
            if first_line == "":
                raise ValueError("Input file is empty.")
            nb_jobs=int(first_line)
            # in each line, the first number is the processing time, the second number is the due date, the third number is the weight
            processing_times=[0]*nb_jobs
            weights=[0]*nb_jobs
            due_dates = [0]*nb_jobs
            job_idx = 0
            for line in f:
                stripped = line.strip()
                if stripped == "":
                    continue
                values = stripped.split()
                if len(values) < 3:
                    raise ValueError(f"Invalid job line: '{stripped}'. Expected 3 values.")
                if job_idx >= nb_jobs:
                    raise ValueError("File contains more jobs than declared in the first line.")
                processing_times[job_idx] = int(values[0])
                due_dates[job_idx] = int(values[1])
                weights[job_idx] = int(values[2])
                job_idx += 1

            if job_idx != nb_jobs:
                raise ValueError(
                    f"File contains {job_idx} jobs but first line declares {nb_jobs}."
                )

            return SchedulingInstance(nb_jobs, processing_times, weights, due_dates)




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
    # numpy array much faster than list comprehension
    if isinstance(pi, np.ndarray):
        return np.maximum(pi, 0)
    
    # compherension list slow
    return [max(x, 0) for x in pi]

# Returns the new step size
def update_step_size(step_size):
    # choix arbitraire 0.8 ; should be decreasing and not too fast
    return 0.8 * step_size

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

##############################################
#          Basic subgradient procedure       #
##############################################
def basic_subgradient(initial_pi, initial_mu, min_step_size):
    pi = initial_pi
    mu = initial_mu

    step_size = 2.0  # initial step

    best_Dualvalue = -math.inf
    best_x = None

    history = []

    while step_size > min_step_size:
        # solve Lagrangian subproblem
        dual_value, x = compute_dual_function(pi, mu)

        # keep best
        if dual_value > best_Dualvalue:
            best_Dualvalue = dual_value
            best_x = x

        # subgradient
        sg_pi, sg_mu = compute_subgradient(x)

        # update
        pi = pi + step_size * sg_pi 
        mu = mu + step_size * sg_mu

        # projection pi >= 0
        pi = project_solution(pi) # returns compherension list

        # update step
        step_size = update_step_size(step_size)

    return round(best_Dualvalue, 2), best_x

##############################################################
#          Subgradient with Polyak step size procedure       #
##############################################################
def subgradientPolyak(initial_pi, initial_mu, min_step_size):
    pi = initial_pi
    mu = initial_mu

    step_size = 2.0  # initial step
    beta_k = 1.0 
    # if beta_k is constant then the method is not guaranteed to converge TAKES to LONG
    # beta_k will be updated down

    L_star = compute_objective_function(feasible_x_sol()) # primal bound

    best_Dualvalue = -math.inf
    best_x = None

    while step_size > min_step_size:
        # solve Lagrangian subproblem
        dual_value, x = compute_dual_function(pi, mu)

        # keep best
        if dual_value > best_Dualvalue:
            best_Dualvalue = dual_value
            best_x = x

        # subgradient
        sg_pi, sg_mu = compute_subgradient(x)

        # update
        pi = pi + step_size * sg_pi 
        mu = mu + step_size * sg_mu

        # projection pi >= 0
        pi = np.array(project_solution(pi))

        # update step
        step_size = update_polyak_step_size(beta_k, L_star, dual_value, np.concatenate([sg_pi, sg_mu]))

    return round(best_Dualvalue, 2), best_x


##############################################
#     Deflected subgradient procedure: ADS   #
##############################################
def compute_direction_ADS(sg_pi, sg_mu, direction_pi, direction_mu):
    # complete the code
    pass


def subgradient_ADS(initial_pi, initial_mu, min_step_size):
    # complete the code
    pass



##############################################
#                Cutting planes              #
##############################################
def initialize_master_program(nb_ineq_ctrs, nb_eq_ctrs, initial_x_sol):
    # complete the code
    pass

def cutting_planes(epsilon):
    # complete the code
    pass
