# hopfield/network.py
import numpy as np
from config.system_config import SystemConfig

class HopfieldNetwork:
    def __init__(self, size=None):
        params = SystemConfig.get_hopfield_params()
        self.size = size or params['size']
        self.threshold = params['threshold']
        self.max_iter = params['max_iter']
        self.weights = np.zeros((self.size, self.size))
        self.patterns = []
    
    def train(self, patterns):
        self.patterns = [self._binarize(p) for p in patterns]
        n_patterns = len(self.patterns)
        
        self.weights = np.zeros((self.size, self.size))
        for pattern in self.patterns:
            self.weights += np.outer(pattern, pattern)
        
        self.weights /= n_patterns
        np.fill_diagonal(self.weights, 0)
    
    def _binarize(self, pattern):
        if len(pattern) != self.size:
            pattern = self._resize_pattern(pattern)
        return np.where(np.array(pattern) >= 0, 1, -1)
    
    def _resize_pattern(self, pattern):
        if len(pattern) > self.size:
            return pattern[:self.size]
        else:
            padded = list(pattern) + [0] * (self.size - len(pattern))
            return padded
    
    def recall(self, input_pattern, async_update=True):
        state = self._binarize(input_pattern)
        
        for iteration in range(self.max_iter):
            if async_update:
                new_state = self._async_update(state)
            else:
                new_state = self._sync_update(state)
            
            if np.array_equal(state, new_state):
                break
            state = new_state
        
        return state
    
    def _async_update(self, state):
        new_state = state.copy()
        indices = np.random.permutation(self.size)
        
        for i in indices:
            activation = np.dot(self.weights[i], new_state) - self.threshold
            new_state[i] = 1 if activation >= 0 else -1
        
        return new_state
    
    def _sync_update(self, state):
        activations = np.dot(self.weights, state) - self.threshold
        return np.where(activations >= 0, 1, -1)
    
    def energy(self, state):
        return -0.5 * np.dot(state, np.dot(self.weights, state)) + np.sum(state * self.threshold)
    
    def pattern_similarity(self, pattern1, pattern2):
        return np.dot(pattern1, pattern2) / self.size

# hopfield/pattern_storage.py
import numpy as np
from .network import HopfieldNetwork

class PatternStorage:
    def __init__(self, network_size=64):
        self.network = HopfieldNetwork(network_size)
        self.stored_patterns = {}
        self.pattern_ids = []
    
    def store_pattern(self, pattern_id, pattern_data):
        binary_pattern = self._convert_to_binary(pattern_data)
        self.stored_patterns[pattern_id] = binary_pattern
        self.pattern_ids.append(pattern_id)
        self._retrain_network()
        return True
    
    def retrieve_pattern(self, input_data, threshold=0.8):
        binary_input = self._convert_to_binary(input_data)
        recalled = self.network.recall(binary_input)
        
        best_match = None
        best_similarity = -1
        
        for pattern_id, stored_pattern in self.stored_patterns.items():
            similarity = self.network.pattern_similarity(recalled, stored_pattern)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = pattern_id
        
        return best_match, best_similarity
    
    def _convert_to_binary(self, data):
        if isinstance(data, str):
            data = [ord(c) for c in data]
        elif isinstance(data, bytes):
            data = list(data)
        
        # Normalize to network size
        if len(data) > self.network.size:
            data = data[:self.network.size]
        elif len(data) < self.network.size:
            data.extend([0] * (self.network.size - len(data)))
        
        # Convert to bipolar
        mean_val = np.mean(data)
        return [1 if x >= mean_val else -1 for x in data]
    
    def _retrain_network(self):
        patterns = list(self.stored_patterns.values())
        self.network.train(patterns)

# hopfield/authenticator.py
from .pattern_storage import PatternStorage
import hashlib

class HopfieldAuthenticator:
    def __init__(self, network_size=64):
        self.pattern_storage = PatternStorage(network_size)
        self.auth_threshold = 0.7
    
    def register_user(self, user_id, biometric_data):
        pattern_hash = hashlib.sha256(biometric_data.encode()).hexdigest()
        success = self.pattern_storage.store_pattern(user_id, pattern_hash)
        return success
    
    def authenticate_user(self, biometric_data):
        pattern_hash = hashlib.sha256(biometric_data.encode()).hexdigest()
        user_id, similarity = self.pattern_storage.retrieve_pattern(
            pattern_hash, self.auth_threshold
        )
        
        if user_id:
            return {"authenticated": True, "user_id": user_id, "confidence": similarity}
        else:
            return {"authenticated": False, "user_id": None, "confidence": 0.0}
    
    def get_authentication_mask(self, user_id):
        if user_id in self.pattern_storage.stored_patterns:
            pattern = self.pattern_storage.stored_patterns[user_id]
            # Convert bipolar to bytes
            mask = bytearray()
            for i in range(0, len(pattern), 8):
                byte_val = 0
                for j in range(8):
                    if i + j < len(pattern) and pattern[i + j] == 1:
                        byte_val |= (1 << j)
                mask.append(byte_val)
            return bytes(mask)
        return None