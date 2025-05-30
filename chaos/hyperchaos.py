# chaos/hyperchaos.py
import numpy as np
from scipy.integrate import solve_ivp
from config.system_config import SystemConfig

class HyperchaosSystem:
    def __init__(self, k1=None, k2=None, k3=None):
        params = SystemConfig.get_chaos_params()
        self.k1 = k1 or params['k1']
        self.k2 = k2 or params['k2'] 
        self.k3 = k3 or params['k3']
        self.dt = params['dt']
    
    def equations(self, t, state):
        x, y, w, u, v = state
        dx = 10 * (y - x) + u
        dy = 28 * x - y - x * (w**2) - v
        dw = self.k1 * x * y * w - self.k2 * w + self.k3 * v
        du = -x * (w**2) + 2 * u
        dv = 8 * y
        return [dx, dy, dw, du, dv]
    
    def generate_trajectory(self, initial_state, t_span, num_points):
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        solution = solve_ivp(self.equations, t_span, initial_state, 
                           t_eval=t_eval, method='RK45')
        return solution.y
    
    def lyapunov_exponent(self, initial_state, time_span=100, dt=0.01):
        n_steps = int(time_span / dt)
        state = np.array(initial_state, dtype=float)
        perturbation = np.random.normal(0, 1e-8, 5)
        
        lyap_sum = 0.0
        for _ in range(n_steps):
            # Integrate original system
            k1 = dt * np.array(self.equations(0, state))
            k2 = dt * np.array(self.equations(0, state + k1/2))
            k3 = dt * np.array(self.equations(0, state + k2/2))  
            k4 = dt * np.array(self.equations(0, state + k3))
            state += (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Integrate perturbed system
            perturbed_state = state + perturbation
            k1_p = dt * np.array(self.equations(0, perturbed_state))
            k2_p = dt * np.array(self.equations(0, perturbed_state + k1_p/2))
            k3_p = dt * np.array(self.equations(0, perturbed_state + k2_p/2))
            k4_p = dt * np.array(self.equations(0, perturbed_state + k3_p))
            perturbed_state += (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6
            
            # Calculate divergence
            separation = perturbed_state - state
            distance = np.linalg.norm(separation)
            
            if distance > 0:
                lyap_sum += np.log(distance / 1e-8)
                perturbation = separation * (1e-8 / distance)
        
        return lyap_sum / (n_steps * dt)

# chaos/key_generator.py  
import hashlib
from .hyperchaos import HyperchaosSystem

class ChaosKeyGenerator:
    def __init__(self):
        self.chaos = HyperchaosSystem()
    
    def generate_initial_state(self, seed):
        if isinstance(seed, str):
            seed = seed.encode()
        hash_bytes = hashlib.sha256(seed).digest()
        
        initial_state = []
        for i in range(0, 20, 4):
            value = int.from_bytes(hash_bytes[i:i+4], byteorder='big')
            normalized = (value / (2**32 - 1)) * 2 - 1
            initial_state.append(normalized)
        
        return initial_state
    
    def generate_keystream(self, seed, length):
        initial_state = self.generate_initial_state(seed)
        trajectory = self.chaos.generate_trajectory(
            initial_state, (0, length * 0.01), length + 100
        )
        
        # Skip transient and extract keystream
        x_vals = trajectory[0, 100:][:length]
        y_vals = trajectory[1, 100:][:length]
        
        keystream = bytearray()
        for i in range(length):
            value = int(abs((x_vals[i] + y_vals[i]) * 1000) % 256)
            keystream.append(value)
        
        return bytes(keystream)
    
    def entropy_measure(self, data):
        if not data:
            return 0.0
        
        freq = np.bincount(data) / len(data)
        freq = freq[freq > 0]
        return -np.sum(freq * np.log2(freq))