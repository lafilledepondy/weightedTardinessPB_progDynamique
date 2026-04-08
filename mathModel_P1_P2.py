# ##############################################
# # Functions defining mathematical program P1 #
# ##############################################

# number of elements of the mathematical program
# returns nb_variables, nb_ineq_ctrs, nb_eq_ctrs
def math_prog_dims():
    return 2,2,0

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
def math_prog_dims():
    return 2,1,0

# returns the value of the objective function at x
def compute_objective_function(x):
    return (x[0]-2)*(x[0]-2)+0.25*x[1]*x[1]

# returns the value of the functions defining the inequality constraints at x
def compute_ineq_ctrs_functions(x):
    return numpy.array([ x[0]-3.5*x[1]-1 ])

# returns the value of the functions defining the equality constraints at x
def compute_eq_ctrs_functions(x):
    return numpy.array([])

# compute the value of the dual function for given multipliers 
# pi (for inequality constraints) and mu (for equality constraints)
# returns the value, and an optimal solution of the Lagrange subproblem
def compute_dual_function(pi,mu):
    optimal_sol=[2-1.5*pi[0],pi[0]]
    return -2.5*pi[0]*pi[0]+pi[0], optimal_sol

# returns an arbitrary feasible solution
def feasible_x_sol():
    return [0,4/3]
# ##################################
# # End of mathematical program P2 #
# ##################################