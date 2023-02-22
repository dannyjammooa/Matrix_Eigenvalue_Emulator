import numpy as np
import concurrent.futures
import schedule
import fun
from scipy.optimize import minimize

def gradient_tempering(y_true, dt, n, MM_num_coeff, temperatures, T0, eta,beta, iters,matrix_type):
    num_temps = len(temperatures)
    current_temp_idx = 0
    initial_coeff = np.random.randn(MM_num_coeff)
    initial_cost = fun.cost_func(initial_coeff, y_true, dt, n,matrix_type)
    best_coeff = initial_coeff
    best_cost = initial_cost
    current_coeff = initial_coeff
    current_cost = initial_cost
    optimizer_scheduler = schedule.Adam(eta,rho=0.9, rho2=0.999)
    AE = fun.AE_mat(n)
    BE = fun.BE_mat(n)
    for i in range(iters):
        T = temperatures[current_temp_idx]
        scale = beta * (np.sqrt(T) / np.sqrt(T0))
        random_change = scale * np.random.normal(MM_num_coeff)
        A_grad,B_grad = fun.grad_cost_func(current_coeff,y_true,dt,n,matrix_type,AE,BE)
        grad_change = fun.update(A_grad,B_grad,optimizer_scheduler,n)
        proposed_coeff = current_coeff -grad_change + random_change
        proposed_cost = fun.cost_func(proposed_coeff, y_true, dt, n,matrix_type)
        dE = proposed_cost - current_cost
        if proposed_cost < current_cost or np.random.rand() < np.exp(-dE / T):
            current_cost = proposed_cost
            current_coeff = proposed_coeff
        if proposed_cost < best_cost:
            current_cost = proposed_cost
            best_cost = proposed_cost
            current_coeff = proposed_coeff
            best_coeff = proposed_coeff
        current_temp_idx = (current_temp_idx + np.random.randint(1, num_temps)) % num_temps
        if np.random.rand() < np.exp((current_cost - proposed_cost) / T):
            current_temp_idx = (current_temp_idx + np.random.randint(1, num_temps)) % num_temps
    optimizer_scheduler.reset()
    res = minimize(fun=fun.cost_func, x0=best_coeff,args=(y_true,dt,n,matrix_type), method='Nelder-Mead')
    return res.x, res.fun

def gradient_tempering2(y_true, dt, n, MM_num_coeff, temperatures, T0, eta,beta, iters,matrix_type):
    num_temps = len(temperatures)
    current_temp_idx = 0
    initial_coeff = MM_num_coeff
    initial_cost = fun.cost_func(initial_coeff, y_true, dt, n,matrix_type)
    best_coeff = initial_coeff
    best_cost = initial_cost
    current_coeff = initial_coeff
    current_cost = initial_cost
    optimizer_scheduler = schedule.Adam(eta,rho=0.9, rho2=0.999)
    AE = fun.AE_mat(n)
    BE = fun.BE_mat(n)
    for i in range(iters):
        T = temperatures[current_temp_idx]
        scale = beta * (np.sqrt(T) / np.sqrt(T0))
        random_change = scale * np.random.normal(len(MM_num_coeff))
        A_grad,B_grad = fun.grad_cost_func(current_coeff,y_true,dt,n,matrix_type,AE,BE)
        grad_change = fun.update(A_grad,B_grad,optimizer_scheduler,n)
        proposed_coeff = current_coeff -grad_change + random_change
        proposed_cost = fun.cost_func(proposed_coeff, y_true, dt, n,matrix_type)
        dE = proposed_cost - current_cost
        if proposed_cost < current_cost or np.random.rand() < np.exp(-dE / T):
            current_cost = proposed_cost
            current_coeff = proposed_coeff
        if proposed_cost < best_cost:
            current_cost = proposed_cost
            best_cost = proposed_cost
            current_coeff = proposed_coeff
            best_coeff = proposed_coeff
        current_temp_idx = (current_temp_idx + np.random.randint(1, num_temps)) % num_temps
        if np.random.rand() < np.exp((current_cost - proposed_cost) / T):
            current_temp_idx = (current_temp_idx + np.random.randint(1, num_temps)) % num_temps
    optimizer_scheduler.reset()
    res = minimize(fun=fun.cost_func, x0=best_coeff,args=(y_true,dt,n,matrix_type), method='Nelder-Mead')
    return res.x, res.fun

def parallel_annealing(y_true, dt, n, num_coeff, initial_temp, final_temp,eta, beta, iters,matrix_type):
    # Generate temperature schedules for each worker
    temp_schedules = schedule.TemperatureSchedules()
    temperature_schedules = temp_schedules.generate_temperature_schedules(initial_temp, final_temp, iters)
    # Initialize list to hold results
    results = []
    # Use a ProcessPoolExecutor to run the tempering function in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # Submit a tempering task for each temperature schedule
        tasks = [executor.submit(gradient_tempering, y_true, dt, n, num_coeff, T, initial_temp,eta, beta, iters,matrix_type) for T in temperature_schedules]
        # Collect the results as they are completed
        for f in concurrent.futures.as_completed(tasks):
            results.append(f.result())
    # Clean and return the results
    return fun.cleaning(results)

def random_initial(y_true, dt, n, num_coeff, initial_temp, final_temp,eta, beta, iters,matrix_type, epoch):
    # Initialize list to hold results
    results = []
    # Use a ProcessPoolExecutor to run the parallel_annealing function in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit a parallel_annealing task for the specified number of epochs
        tasks = [executor.submit(parallel_annealing, y_true, dt, n, num_coeff, initial_temp, final_temp,eta, beta, iters,matrix_type) for _ in range(epoch)]
        # Collect the results as they are completed
        for f in concurrent.futures.as_completed(tasks):
            results.append(f.result())
    # Clean and return the results
    return fun.cleaning(results)