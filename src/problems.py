import numpy as np

##############################################
# Functions defining mathematical program P1 #
##############################################
def p1_math_prog_dims():
    return 2, 2, 0  # 2 variables, 2 inequality ctrs, 0 equality ctrs


def p1_compute_objective_function(x):
    return x[0] - 4 * x[1]


def p1_compute_ineq_ctrs_functions(x):
    return np.array([-x[0] - x[1] + 2, x[1] - 1])


def p1_compute_eq_ctrs_functions(x):
    return np.array([])


def p1_compute_dual_function(pi, mu):
    lagrange_costs = [1 - pi[0], -4 - pi[0] + pi[1]]
    optimal_sol = [(3 if v < 0 else 0) for v in lagrange_costs]
    return np.dot(optimal_sol, lagrange_costs) + np.dot([2, -1], pi), optimal_sol


def p1_feasible_x_sol():
    return [3, 1]


##############################################
# Functions defining mathematical program P2 #
##############################################
def p2_math_prog_dims():
    return 2, 1, 0


def p2_compute_objective_function(x):
    return (x[0] - 2) * (x[0] - 2) + 0.25 * x[1] * x[1]


def p2_compute_ineq_ctrs_functions(x):
    return np.array([x[0] - 3.5 * x[1] - 1])


def p2_compute_eq_ctrs_functions(x):
    return np.array([])


def p2_compute_dual_function(pi, mu):
    optimal_sol = [2 - 1.5 * pi[0], pi[0]]
    return -2.5 * pi[0] * pi[0] + pi[0], optimal_sol


def p2_feasible_x_sol():
    return [0, 4 / 3]

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

#################################################################
# Functions defining mathematical program P3=Weighted Tardiness #
#################################################################
def p3_math_prog_dims(instance: SchedulingInstance):
    

    I= list(range(instance.nb_jobs))
    T= list(range(instance.horizon+1))
    Ti={ i: list(range(instance.horizon -instance.processing_times[i]+1)) for i in I }
    return I*Ti, T, I #number of variables I*Ti, number of inequality ctrs T, number of equality ctrs I

def p3_cost(instance: SchedulingInstance,i, t):
    p_i=instance.processing_times[i]
    d_i=instance.due_dates[i]
    w_i=instance.weights[i]
    return w_i * max(0, t + p_i - d_i) #t+pi-di <0 pas de retard

def p3_compute_objective_function(instance: SchedulingInstance, Ti, x):
    obj=0
    for i in range(instance.nb_jobs):
        for t in Ti[i]:
            if x[i][t] > 0:
                obj += p3_cost(instance, i, t) * x[i][t]
    return obj
    
def p3_compute_eq_ctrs_functions(instance: SchedulingInstance, Ti, x):
    return [sum (x[i][t] for t in Ti[i]) - 1 for i in range(instance.nb_jobs)]
        
def p3_compute_ineq_ctrs_functions(instance: SchedulingInstance, Ti, x):
    load=0 *(instance.horizon+1)
    for i in range(instance.nb_jobs):
            p_i = instance.processing_times[i]
            for t_prime in Ti[i]:
                if x[i][t_prime] > 0:
                    for s in range(t_prime, t_prime + p_i):
                        if s <= instance.horizon:
                            load[s] += x[i][t_prime]
    return [l-1 for l in load]
                           

def p3_compute_dual_function(instance: SchedulingInstance, Ti, mu):
    pass

def p3_feasible_x_sol(instance: SchedulingInstance, Ti):
    x={i:{t: 0 for t in Ti[i]} for i in range(instance.nb_jobs)}
    current_time=0
    for i in range(instance.nb_jobs):
        t=current_time
        if t not in Ti[i]:
            t= Ti[i][0]
        x[i][t]=1
        current_time += instance.processing_times[i]
    return x

############################################
# Redirectly to their correcponding models #
############################################
PROBLEMS = {
    "P1": {
        "math_prog_dims": p1_math_prog_dims,
        "compute_objective_function": p1_compute_objective_function,
        "compute_ineq_ctrs_functions": p1_compute_ineq_ctrs_functions,
        "compute_eq_ctrs_functions": p1_compute_eq_ctrs_functions,
        "compute_dual_function": p1_compute_dual_function,
        "feasible_x_sol": p1_feasible_x_sol,
    },
    "P2": {
        "math_prog_dims": p2_math_prog_dims,
        "compute_objective_function": p2_compute_objective_function,
        "compute_ineq_ctrs_functions": p2_compute_ineq_ctrs_functions,
        "compute_eq_ctrs_functions": p2_compute_eq_ctrs_functions,
        "compute_dual_function": p2_compute_dual_function,
        "feasible_x_sol": p2_feasible_x_sol,
    },
    "P3": {
        "math_prog_dims": p3_math_prog_dims,
        "compute_objective_function": p3_compute_objective_function,
        "compute_ineq_ctrs_functions": p3_compute_ineq_ctrs_functions,
        "compute_eq_ctrs_functions": p3_compute_eq_ctrs_functions,
        "compute_dual_function": p3_compute_dual_function,
        "feasible_x_sol": p3_feasible_x_sol,    
    }
}


ACTIVE_PROBLEM = ""

def set_active_problem(problem_name):
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown problem '{problem_name}'. Available: {list(PROBLEMS)}")
    global ACTIVE_PROBLEM
    ACTIVE_PROBLEM = problem_name

def get_active_problem():
    return PROBLEMS[ACTIVE_PROBLEM]

def math_prog_dims():
    return get_active_problem()["math_prog_dims"]()

def compute_objective_function(x):
    return get_active_problem()["compute_objective_function"](x)

def compute_ineq_ctrs_functions(x):
    return get_active_problem()["compute_ineq_ctrs_functions"](x)

def compute_eq_ctrs_functions(x):
    return get_active_problem()["compute_eq_ctrs_functions"](x)

def compute_dual_function(pi, mu):
    return get_active_problem()["compute_dual_function"](pi, mu)

def feasible_x_sol():
    return get_active_problem()["feasible_x_sol"]()