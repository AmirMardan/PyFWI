import copy
import logging
import numpy as np



def linesearch(fun, fprime, xk, pk, gk=None, fval_old=None, f_max=50, alpha0=None, args=()):

    x0 = copy.deepcopy(xk)
    rho = 0.5
    rho_inc = 0.5

    # Defining the initial alpha based on the ratio between data and gradient
    if alpha0 is None:
        alpha0 = np.abs(xk).max() / np.abs(pk).max()  # 1000.0

    initial_alpha = np.copy(alpha0)  # for comparing the alpha as a condition of increasing alpha
    fc = [0]
    max_call = 15
    
    def phi(alpha):
        fc[0] += 1
        x1 = x0 + alpha * pk
        return fun(x1, *args)
    
    def dephi(alpha):
        x1 = x0 + alpha * pk
        return fprime(x1)
    
    if fval_old is None:
        fval_old = phi(0.0)
    
    fval_new = phi(alpha0)
    
    count = 0

    # For decreasing the alpha
    while (np.isnan(fval_new) or (fval_new > fval_old)):  # & (count < max_call):
        alpha0 *= rho
        fval_new = phi(alpha0)
        
        print(f"{alpha0 = } .......... {fval_new = :.4f} .......... {fval_old = :.4f}")
        count += 1

    if count == 0: # If we need to increase the alpha
        while (fval_new < fval_old) & (count < max_call):
            alpha_inc = alpha0 + rho_inc * alpha0
            fval_new_inc = phi(alpha_inc)

            count += 1
            print(f"{alpha_inc = } .......... {fval_new_inc = :.4f} .......... {fval_old = :.4f}")
            if fval_new_inc < fval_new:
                alpha0 = np.copy(alpha_inc)
                fval_new = np.copy(fval_new_inc)
            else:
                break

    if (fval_new > fval_old) & (count == max_call):
        alpha = None
        logging.warning("Linesearch didn't converge.")
    else:
        alpha = alpha0 
    print(f'{initial_alpha = } -------------------------------{alpha = :.4f} with  {count = }')
    
    return alpha, phi(alpha), dephi(alpha)
        
        