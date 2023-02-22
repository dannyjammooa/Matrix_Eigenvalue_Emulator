import numpy as np
import concurrent.futures
from scipy.optimize import minimize
import schedule
import fun

def tempering(y_true, dt, n, MM_num_coeff, temperatures, T0, beta, iters,matrix_type):
    # Compute number of temperatures in schedule
    num_temps = len(temperatures)
    # Initialize current temperature index
    current_temp_idx = 0
    # Generate initial coefficients using random normal distribution
    initial_coeff = np.random.randn(MM_num_coeff)
    # Compute initial cost function value
    initial_cost = fun.cost_func(initial_coeff, y_true, dt, n,matrix_type)
    # Set best coefficients and cost function value to initial values
    best_coeff = initial_coeff
    best_cost = initial_cost
    # Set current coefficients and cost function value to initial values
    current_coeff = initial_coeff
    current_cost = initial_cost
    # Loop through iterations
    for i in range(iters):
        # Get current temperature from schedule
        T = temperatures[current_temp_idx]
        # Compute scale for temperature change
        scale = beta * (np.sqrt(T) / np.sqrt(T0))
        # Generate new coefficients using random normal distribution
        proposed_coeff = current_coeff + scale * np.random.normal(MM_num_coeff)
        # Compute cost function value for new coefficients
        proposed_cost = fun.cost_func(proposed_coeff, y_true, dt, n,matrix_type)
        # Compute change in cost function value
        dE = proposed_cost - current_cost
        # Check if new coefficients are better than current coefficients 
        # or if a random number is better then then the Boltzmann distribution
        if proposed_cost < current_cost or np.random.rand() < np.exp(-dE / T):
            # Set current coefficients and cost function value to proposed values
            current_cost = proposed_cost
            current_coeff = proposed_coeff
        # Check if new coefficients are better than best coefficients
        if proposed_cost < best_cost:
            # Set current and best coefficients and cost function values to proposed values
            current_cost = proposed_cost
            best_cost = proposed_cost
            current_coeff = proposed_coeff
            best_coeff = proposed_coeff
        # Update the current temperature index using simulated tempering
        current_temp_idx = (current_temp_idx + np.random.randint(1, num_temps)) % num_temps
        # Occasionally update the temperature index even if the energy doesn't change
        if np.random.rand() < np.exp((current_cost - proposed_cost) / T):
            current_temp_idx = (current_temp_idx + np.random.randint(1, num_temps)) % num_temps
    # Use the Nelder-Mead method to further optimize the best coefficients
    res = minimize(fun=fun.cost_func, x0=best_coeff,args=(y_true,dt,n,matrix_type), method='Nelder-Mead')
    return res.x, res.fun

def tempering2(y_true, dt, n, MM_num_coeff, temperatures, T0, beta, iters,matrix_type):
    # Compute number of temperatures in schedule
    num_temps = len(temperatures)
    # Initialize current temperature index
    current_temp_idx = 0
    # Generate initial coefficients using random normal distribution
    initial_coeff = MM_num_coeff
    # Compute initial cost function value
    initial_cost = fun.cost_func(initial_coeff, y_true, dt, n,matrix_type)
    # Set best coefficients and cost function value to initial values
    best_coeff = initial_coeff
    best_cost = initial_cost
    # Set current coefficients and cost function value to initial values
    current_coeff = initial_coeff
    current_cost = initial_cost
    # Loop through iterations
    for i in range(iters):
        # Get current temperature from schedule
        T = temperatures[current_temp_idx]
        # Compute scale for temperature change
        scale = beta * (np.sqrt(T) / np.sqrt(T0))
        # Generate new coefficients using random normal distribution
        proposed_coeff = current_coeff + scale * np.random.normal(len(MM_num_coeff))
        # Compute cost function value for new coefficients
        proposed_cost = fun.cost_func(proposed_coeff, y_true, dt, n,matrix_type)
        # Compute change in cost function value
        dE = proposed_cost - current_cost
        # Check if new coefficients are better than current coefficients 
        # or if a random number is better then then the Boltzmann distribution
        if proposed_cost < current_cost or np.random.rand() < np.exp(-dE / T):
            # Set current coefficients and cost function value to proposed values
            current_cost = proposed_cost
            current_coeff = proposed_coeff
        # Check if new coefficients are better than best coefficients
        if proposed_cost < best_cost:
            # Set current and best coefficients and cost function values to proposed values
            current_cost = proposed_cost
            best_cost = proposed_cost
            current_coeff = proposed_coeff
            best_coeff = proposed_coeff
        # Update the current temperature index using simulated tempering
        current_temp_idx = (current_temp_idx + np.random.randint(1, num_temps)) % num_temps
        # Occasionally update the temperature index even if the energy doesn't change
        if np.random.rand() < np.exp((current_cost - proposed_cost) / T):
            current_temp_idx = (current_temp_idx + np.random.randint(1, num_temps)) % num_temps
    # Use the Nelder-Mead method to further optimize the best coefficients
    res = minimize(fun=fun.cost_func, x0=best_coeff,args=(y_true,dt,n,matrix_type), method='Nelder-Mead')
    return res.x, res.fun

def parallel_annealing(y_true, dt, n, num_coeff, initial_temp, final_temp, beta, iters,matrix_type):
    # Generate temperature schedules for each worker
    temp_schedules = schedule.TemperatureSchedules()
    temperature_schedules = temp_schedules.generate_temperature_schedules(initial_temp, final_temp, iters)
    # Initialize list to hold results
    results = []
    # Use a ProcessPoolExecutor to run the tempering function in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # Submit a tempering task for each temperature schedule
        tasks = [executor.submit(tempering, y_true, dt, n, num_coeff, T, initial_temp, beta, iters,matrix_type) for T in temperature_schedules]
        # Collect the results as they are completed
        for f in concurrent.futures.as_completed(tasks):
            results.append(f.result())
    # Clean and return the results
    return fun.cleaning(results)

def random_initial(y_true, dt, n, num_coeff, initial_temp, final_temp, beta, iters,matrix_type, epoch):
    # Initialize list to hold results
    results = []
    # Use a ProcessPoolExecutor to run the parallel_annealing function in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit a parallel_annealing task for the specified number of epochs
        tasks = [executor.submit(parallel_annealing, y_true, dt, n, num_coeff, initial_temp, final_temp, beta, iters,matrix_type) for _ in range(epoch)]
        # Collect the results as they are completed
        for f in concurrent.futures.as_completed(tasks):
            results.append(f.result())
    # Clean and return the results
    return fun.cleaning(results)