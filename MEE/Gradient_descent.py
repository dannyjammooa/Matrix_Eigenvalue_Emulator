import numpy as np
import concurrent.futures
import schedule
import fun

#gradient descent
def gradient_descent(y_true, dt, n, MM_num_coeff,optimizer, eta,iters):
    best_coeff = np.random.randn(MM_num_coeff)
    best_cost = fun.cost_func(best_coeff, y_true, dt, n)
    current_coeff = best_coeff
    current_cost = best_cost
    optimizer_scheduler = optimizer
    for i in range(iters):
        A_grad,B_grad = fun.grad_cost_func(current_coeff,y_true,dt,n)
        proposed_coeff = current_coeff -fun.update(A_grad,B_grad,optimizer_scheduler,n)
        proposed_cost = fun.cost_func(proposed_coeff, y_true, dt, n)
        if proposed_cost < current_cost:
            current_cost = proposed_cost
            current_coeff = proposed_coeff
        if proposed_cost < best_cost:
            best_coeff = proposed_coeff
            best_cost = proposed_cost
    optimizer_scheduler.reset()
    return best_coeff, best_cost

def gradient_descent2(y_true, dt, n, MM_num_coeff,optimizer, eta,iters):
    best_coeff = MM_num_coeff
    best_cost = fun.cost_func(best_coeff, y_true, dt, n)
    current_coeff = best_coeff
    current_cost = best_cost
    optimizer_scheduler = optimizer
    for i in range(iters):
        A_grad,B_grad = fun.grad_cost_func(current_coeff,y_true,dt,n)
        proposed_coeff = current_coeff -fun.update(A_grad,B_grad,optimizer_scheduler,n)
        proposed_cost = fun.cost_func(proposed_coeff, y_true, dt, n)
        if proposed_cost < current_cost:
            current_cost = proposed_cost
            current_coeff = proposed_coeff
        if proposed_cost < best_cost:
            best_coeff = proposed_coeff
            best_cost = proposed_cost
    optimizer_scheduler.reset()
    return best_coeff, best_cost

def parallel_optimizers(y_true, dt, n, MM_num_coeff, eta,iters):
    # Generate schedule for each worker
    momentum = schedule.Momentum(eta,momentum =0.9)
    adagrad = schedule.Adagrad(eta)
    adagradmomentum = schedule.AdagradMomentum(eta,momentum =0.9)
    rms_prop = schedule.RMS_prop(eta,rho=0.9)
    adam  =    schedule.Adam(eta,rho=0.9, rho2=0.999)
    optimizers = schedule.OptimizerSchedules([momentum, adagrad,adagradmomentum,rms_prop,adam])
    # Initialize list to hold results
    results = []
    # Use a ProcessPoolExecutor to run the gradient_descent function in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # Submit a gradient_descent task for each optimizer schedule
        tasks = [executor.submit(gradient_descent, y_true, dt, n, MM_num_coeff,optimizer, eta,iters) for optimizer in optimizers]
        # Collect the results as they are completed
        for f in concurrent.futures.as_completed(tasks):
            results.append(f.result())
    # Clean and return the results
    return fun.cleaning(results)

def random_initial(y_true, dt, n, MM_num_coeff, eta,iters, epoch):
    # Initialize list to hold results
    results = []
    # Use a ProcessPoolExecutor to run the parallel_optimizers function in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit a parallel_optimizers task for the specified number of epochs
        tasks = [executor.submit(parallel_optimizers, y_true, dt, n, MM_num_coeff, eta,iters) for _ in range(epoch)]
        # Collect the results as they are completed
        for f in concurrent.futures.as_completed(tasks):
            results.append(f.result())
    # Clean and return the results
    return fun.cleaning(results)