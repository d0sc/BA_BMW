import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from get_bounds import get_bounds_for_hri_tolerances
from pyswarm import pso
from model_tol_cost import cost_from_tol_hard_approach
from model_tol_analysis import quality_from_tol
from ML_hri_eol_regression import make_prediction
from joblib import load


## DEVO
def differential_evolution_optimization():
        
    bounds = [(25, 250),
              (25, 250),
              (25, 250),
              (25, 250),
              (25, 250),
              (25, 250),
              (25, 250),
              (25, 250),
              (25, 250),
              (25, 250)]

    nlc1 = NonlinearConstraint(quality_from_tol,
                              50,
                              np.inf)


    result = differential_evolution(cost_from_tol_hard_approach,
                                    bounds,
                                    constraints=(nlc1),
                                    seed=1,
                                    disp=True)

    print(result)
    

## PSO
def particel_swarm_optimization():
    
    lb = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
    ub = [250, 250, 250, 250, 250, 250, 250, 250, 250, 250]

    xopt, fopt = pso(cost_from_tol_hard_approach, 
                     lb=lb, 
                     ub=ub, 
                     f_ieqcons=quality_from_tol, 
                     maxiter=2000,
                     swarmsize=10)
    
    print("################ -- ################\n", \
          "Optimaler Toleranz-vektor:\n", xopt)
    


if __name__ == '__main__':
    particel_swarm_optimization()
    # differential_evolution_optimization()
    
    

