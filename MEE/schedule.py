import numpy as np

#scheduler for simulated annealing
class TemperatureSchedules:
    def exponential_logarithmic_schedule(self, T0, Tn, iters):
        # Compute temperature factor
        temp_factor = -np.log(T0 / Tn)
        # Generate temperature schedule
        temp_schedule = [T0 * np.exp(temp_factor * i / iters) for i in range(iters)]
        # Return temperature schedule
        return temp_schedule

    def linear_schedule(self, T0, Tn, iters):
        # Calculate the slope of the temperature schedule
        slope = (Tn - T0) / iters
        # Create an array of temperatures using a linear schedule
        T = [T0 + slope * i for i in range(iters)]
        # Return the array of temperatures
        return T

    def exponential_schedule(self, T0, Tn, iters):
        # Calculate the rate of decay of the temperature schedule
        rate = np.log(T0 / Tn) / iters
        # Create an array of temperatures using an exponential schedule
        T = [T0 * np.exp(-rate * i) for i in range(iters)]
        # Return the array of temperatures
        return T

    def logarithmic_schedule(self, T0, Tn, iters):
        # Calculate the base of the logarithmic temperature schedule
        base = np.exp((np.log(Tn) - np.log(T0)) / iters)
        # Create an array of temperatures using a logarithmic schedule
        T = [T0 * base ** i for i in range(iters)]
        # Return the array of temperatures
        return T

    def inverse_schedule(self, T0, Tn, iters):
        # Calculate the rate of decay of the temperature schedule
        rate = 1 / iters
        # Create an array of temperatures using an inverse schedule
        T = [1 / (1 / T0 + rate * i) for i in range(iters)]
        # Return the array of temperatures
        return T

    def generate_temperature_schedules(self, T0, Tn, iters):
        # Generate an array of temperatures using different schedules
        T1 = self.linear_schedule(T0, Tn, iters)
        T2 = self.exponential_schedule(T0, Tn, iters)
        T3 = self.logarithmic_schedule(T0, Tn, iters)
        T4 = self.inverse_schedule(T0, Tn, iters)
        T5 = self.exponential_logarithmic_schedule(T0, Tn, iters)
        # Return the array of temperature schedules
        return [T1, T2, T3, T4, T5]

    
#scheduler for gradient descent     
class Scheduler:
    def __init__(self, eta):
        self.eta = eta
    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError
    # overwritten if needed
    def reset(self):
        pass

class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)
    def update_change(self, gradient):
        return self.eta * gradient
    def reset(self):
        pass

class Momentum(Scheduler):
    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0
    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change
    def reset(self):
        pass

class Adagrad(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)
        self.G_t = None
    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))
        self.G_t += gradient @ gradient.T
        G_t_inverse = 1 / (delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1))))
        return self.eta * gradient * G_t_inverse
    def reset(self):
        self.G_t = None

class AdagradMomentum(Scheduler):
    def __init__(self, eta, momentum):
        super().__init__(eta)
        self.G_t = None
        self.momentum = momentum
        self.change = 0
    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))
        self.G_t += gradient @ gradient.T
        G_t_inverse = 1 / (delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1))))
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change
    def reset(self):
        self.G_t = None

class RMS_prop(Scheduler):
    def __init__(self, eta, rho):
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0
    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))
    def reset(self):
        self.second = 0.0
        
class Adam(Scheduler):
    def __init__(self, eta, rho, rho2):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient
        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)
        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self):
        self.n_epochs += 1
        self.moment = 0
        self.second = 0
        
class OptimizerSchedules:
    def __init__(self, optimizers):
        self.optimizers = optimizers
    def __iter__(self):
        return iter(self.optimizers)
    def update_change(self, gradient):
        updates = optimizer.update_change(gradient)
        return updates
    def reset(self):
        for optimizer in self.optimizers:
            optimizer.reset()