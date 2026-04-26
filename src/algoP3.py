import numpy as np
from problems import SchedulingInstance, p3_cost, lagrange_cost


def p3_compute_dual_function_ResolutionPB(instance: SchedulingInstance, Ti, pi, mu):
    dual_val=0
    x_opt= np.zeros((instance.nb_jobs, instance.horizon+1))

    for i in range(instance.nb_jobs):
        p_i=instance.processing_times[i]
        min_cost=float('inf')
        best_t=None

        for t in Ti[i]:
            cost=p3_cost(instance, i, t)
            sum_mu= np.sum(mu[t:t+p_i])
            lagrangien= (cost +sum_mu)* x_opt[i,t]

            if lagrange_cost < min_cost:
                min_cost=lagrange_cost
                best_t=t
        
        x_opt[i,best_t]=1
        dual_val += min_cost
    
    lagrangien -= np.sum(mu) 
    return lagrangien, x_opt