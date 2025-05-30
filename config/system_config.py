# config/system_config.py
import numpy as np

class SystemConfig:
    # Hyperchaotic system parameters
    CHAOS_K1 = 1.0
    CHAOS_K2 = 4.0
    CHAOS_K3 = 1.2
    CHAOS_DT = 0.01
    CHAOS_SKIP = 100
    
    # Hopfield network parameters
    HOPFIELD_SIZE = 64
    HOPFIELD_PATTERNS = 8
    HOPFIELD_THRESHOLD = 0.0
    HOPFIELD_MAX_ITER = 100
    
    # SSO parameters
    SSO_POPULATION = 20
    SSO_GENERATIONS = 50
    SSO_W_MIN = 0.4
    SSO_W_MAX = 0.9
    SSO_C1 = 2.0
    SSO_C2 = 2.0
    SSO_C3 = 1.5
    
    # Crypto parameters
    BLOCK_SIZE = 16
    KEY_SIZE = 32
    IV_SIZE = 16
    
    @classmethod
    def get_chaos_params(cls):
        return {
            'k1': cls.CHAOS_K1,
            'k2': cls.CHAOS_K2, 
            'k3': cls.CHAOS_K3,
            'dt': cls.CHAOS_DT,
            'skip': cls.CHAOS_SKIP
        }
    
    @classmethod
    def get_hopfield_params(cls):
        return {
            'size': cls.HOPFIELD_SIZE,
            'patterns': cls.HOPFIELD_PATTERNS,
            'threshold': cls.HOPFIELD_THRESHOLD,
            'max_iter': cls.HOPFIELD_MAX_ITER
        }
    
    @classmethod
    def get_sso_params(cls):
        return {
            'population': cls.SSO_POPULATION,
            'generations': cls.SSO_GENERATIONS,
            'w_min': cls.SSO_W_MIN,
            'w_max': cls.SSO_W_MAX,
            'c1': cls.SSO_C1,
            'c2': cls.SSO_C2,
            'c3': cls.SSO_C3
        }
    
    @classmethod
    def get_crypto_params(cls):
        return {
            'block_size': cls.BLOCK_SIZE,
            'key_size': cls.KEY_SIZE,
            'iv_size': cls.IV_SIZE
        }