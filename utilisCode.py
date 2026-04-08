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
    pass

# Projects the set of multipliers associated with inequality constraints
# onto the non-negative orthant (returns the vector of multipliers with 
# negative components set to zero).
def project_solution(pi):
    # complete the code
    pass

# Returns the new step size
def update_step_size(step_size):
    # complete the code
    pass

##############################################
#          Basic subgradient procedure       #
##############################################
def basic_subgradient(initial_pi,initial_mu,min_step_size):
    # complete the code
    pass


##############################################
#     Deflected subgradient procedure: ADS   #
##############################################
def compute_direction_ADS(sg_pi,sg_mu,direction_pi,direction_mu):
    # complete the code
    pass


def subgradient_ADS(initial_pi,initial_mu,min_step_size):
    # complete the code
    pass



##############################################
#                Cutting planes              #
##############################################
def initialize_master_program(nb_ineq_ctrs,nb_eq_ctrs,initial_x_sol):
    # complete the code
    pass

def cutting_planes(epsilon):
    # complete the code
    pass