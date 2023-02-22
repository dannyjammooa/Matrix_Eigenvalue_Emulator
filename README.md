# Matrix Eigenvalue Emulator (MEE)

This repository contains code for the Matrix Eigenvale Emulator (MEE).

1.The file fun.py contains some functions such as:
  - eigen, while sorts eigenvalues and eigenvectors from smallest to largest
  - ""_matrix, creats matrix given list of matrix elements and size n
  - Emin and Emin_loop, find the ground state eigenvalue/vector for a given dt
  - numerical_derivative, calculates the numerical derivative of a function and applies Hellmann-Feynman theorem
  - cost_func, calculates the cost function that is minimized
  - grad_cost_func, uses numerical_derivative to calculate the gradient of the cost function
  
2.The file schedule.py contains functions relevent to schedules such as
  - temperature schedules for simulated annealing
  - Optimizer schedules for gradient descent
  
3.The files Simulated_annealing.py, Gradient_descent.py, and SAST_GD.py uses the fun.py and schedule.py to implement the different optimization algorithms.
  - parallel_"", applies an aglorithm with different schedules in parallel
  - random_initial, runs parrallel_"" multiple times in parallel

4. The files toy_model_"" contains notebooks that runs Simulated_annealing.py, Gradient_descent.py, and SAST_GD.py for the system H=H_A+H_B*dt and trains M=A+B*dt
  - However, the "".py files contain code for the system e^(-iHdt)=e^(-iH_Adt)e^(-iH_Bdt)
