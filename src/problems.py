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
    Ti = {i: list(range(instance.horizon - instance.processing_times[i] + 1)) for i in I}
    return sum(len(Ti[i]) for i in I), instance.horizon+1, instance.nb_jobs  # number of variables I*Ti, number of inequality ctrs T, number of equality ctrs I


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


def p3_compute_ineq_ctrs_functions(instance: SchedulingInstance, Ti, x):
    ctr3 = np.zeros(instance.horizon + 1, dtype=float)
    for t in range(instance.horizon + 1):                 
        for i in range(instance.nb_jobs):        
            for t_prime in Ti[i]:           
                if t_prime <= t < t_prime + instance.processing_times[i]:
                    ctr3[t] += x[i][t_prime]
    ctr3 = ctr3 - 1
    return ctr3


def p3_compute_eq_ctrs_functions(instance: SchedulingInstance, Ti, x):
    res=[]
    for i in range(instance.nb_jobs):
        res.append(np.sum(x[i,:]) - 1)
    return np.array(res)                           


def p3_compute_dual_function(instance: SchedulingInstance, Ti, pi, mu):
    # dual_val=0
    x_opt= np.zeros((instance.nb_jobs, instance.horizon+1))

    for i in range(instance.nb_jobs):
        p_i=instance.processing_times[i]
        min_cost=float('inf')
        best_t=None

        for t in Ti[i]:
            cost=p3_cost(instance, i, t)
            sum_mu= np.sum(mu[t:t+p_i])
            lagrangien= (cost +sum_mu)* x_opt[i,t]

            # if lagrange_cost < min_cost:
            #     min_cost=lagrange_cost
            #     best_t=t
        
        # x_opt[i,best_t]=1
        # dual_val += min_cost
    
    lagrangien -= np.sum(mu) 
    return lagrangien, x_opt
            

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


#################################################################
# Functions defining mathematical program P4=Weighted Tardiness #
#################################################################
def p4_math_prog_dims(instance: SchedulingInstance):
    I= list(range(instance.nb_jobs))
    Ti = {i: list(range(instance.horizon - instance.processing_times[i] + 1)) for i in I}
    return sum(len(Ti[i]) for i in I), instance.horizon+1, instance.nb_jobs  # number of variables I*Ti, number of inequality ctrs T, number of equality ctrs I


def p4_cost(instance: SchedulingInstance,i, t):
    p_i=instance.processing_times[i]
    d_i=instance.due_dates[i]
    w_i=instance.weights[i]
    return w_i * max(0, t + p_i - d_i) #t+pi-di <0 pas de retard


def p4_compute_objective_function(instance: SchedulingInstance, Ti, x):
    obj=0
    for i in range(instance.nb_jobs):
        for t in Ti[i]:
            if x[i][t] > 0:
                obj += p4_cost(instance, i, t) * x[i][t]
    return obj

    
def p4_compute_eq_ctrs_functions(instance: SchedulingInstance, Ti, x):
    ctr2 = [sum (x[i][t] for t in Ti[i]) - instance.nb_jobs for i in range(instance.nb_jobs)]
    ctr3 = np.zeros(instance.horizon + 1, dtype=float)
    for t in range(instance.horizon + 1):                 
        for i in range(instance.nb_jobs):        
            for t_prime in Ti[i]:           
                if t_prime <= t < t_prime + instance.processing_times[i]:
                    ctr3[t] += x[i][t_prime]
    ctr3 = ctr3 - 1
    return np.array([ctr2, ctr3])


def p4_compute_ineq_ctrs_functions(instance: SchedulingInstance, Ti, x):
    return np.array([])
                           

def p4_compute_dual_function(instance: SchedulingInstance, Ti, pi, mu):
    x_opt= np.zeros((instance.nb_jobs, instance.horizon+1))

    for i in range(instance.nb_jobs):
        for t in Ti[i]:
            cost=p4_cost(instance, i, t)
            lagrangien = (cost + mu[i]) * x_opt[i,t]
    
    lagrangien -= np.sum(mu) 
    return lagrangien, x_opt


def p4_feasible_x_sol(instance: SchedulingInstance, Ti):
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
    },
    "P4": {
        "math_prog_dims": p4_math_prog_dims,
        "compute_objective_function": p4_compute_objective_function,
        "compute_ineq_ctrs_functions": p4_compute_ineq_ctrs_functions,
        "compute_eq_ctrs_functions": p4_compute_eq_ctrs_functions,
        "compute_dual_function": p4_compute_dual_function,
        "feasible_x_sol": p4_feasible_x_sol,
    }
}


ACTIVE_PROBLEM = ""
ACTIVE_INSTANCE = None
ACTIVE_TI = None

def _compute_time_windows(instance: SchedulingInstance):
    I = list(range(instance.nb_jobs))
    return {i: list(range(instance.horizon - instance.processing_times[i] + 1)) for i in I}


def _require_active_context():
    if ACTIVE_INSTANCE is None:
        raise ValueError("No active scheduling instance set for this problem.")
    if ACTIVE_TI is None:
        raise ValueError("No active time windows (Ti) set for this problem.")
    return ACTIVE_INSTANCE, ACTIVE_TI


def set_active_problem(problem_name, instance: SchedulingInstance | None = None, Ti=None):
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown problem '{problem_name}'. Available: {list(PROBLEMS)}")
    global ACTIVE_PROBLEM, ACTIVE_INSTANCE, ACTIVE_TI
    ACTIVE_PROBLEM = problem_name
    if problem_name in ("P3", "P4") and instance is not None:
        ACTIVE_INSTANCE = instance
        ACTIVE_TI = Ti if Ti is not None else _compute_time_windows(instance)

def get_active_problem():
    return PROBLEMS[ACTIVE_PROBLEM]

def math_prog_dims():
    fn = get_active_problem()["math_prog_dims"]
    if ACTIVE_PROBLEM in ("P3", "P4"):
        instance, _ = _require_active_context()
        return fn(instance)
    return fn()

def compute_objective_function(x):
    fn = get_active_problem()["compute_objective_function"]
    if ACTIVE_PROBLEM in ("P3", "P4"):
        instance, Ti = _require_active_context()
        return fn(instance, Ti, x)
    return fn(x)

def compute_ineq_ctrs_functions(x):
    fn = get_active_problem()["compute_ineq_ctrs_functions"]
    if ACTIVE_PROBLEM in ("P3", "P4"):
        instance, Ti = _require_active_context()
        return fn(instance, Ti, x)
    return fn(x)

def compute_eq_ctrs_functions(x):
    fn = get_active_problem()["compute_eq_ctrs_functions"]
    if ACTIVE_PROBLEM in ("P3", "P4"):
        instance, Ti = _require_active_context()
        return fn(instance, Ti, x)
    return fn(x)

def compute_dual_function(pi, mu):
    fn = get_active_problem()["compute_dual_function"]
    if ACTIVE_PROBLEM in ("P3", "P4"):
        instance, Ti = _require_active_context()
        return fn(instance, Ti, pi, mu)
    return fn(pi, mu)

def feasible_x_sol():
    fn = get_active_problem()["feasible_x_sol"]
    if ACTIVE_PROBLEM in ("P3", "P4"):
        instance, Ti = _require_active_context()
        return fn(instance, Ti)
    return fn()