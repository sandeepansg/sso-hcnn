# Chaos-Hopfield-SSO Cryptographic System

## Overview

This project implements a novel cryptographic framework that combines:
- **Chaos-based cryptography** for key generation and stream ciphers
- **Hopfield Neural Networks** for pattern-based authentication and key validation
- **Shark Smell Optimization (SSO)** for parameter optimization and adaptive security

## Mathematical Foundations

### 1. Hyperchaotic System

The core chaotic system uses a 5D hyperchaotic attractor:

```
dx/dt = 10(y - x) + u
dy/dt = 28x - y - x(w²) - v  
dw/dt = k₁xyw - k₂w + k₃v
du/dt = -x(w²) + 2u
dv/dt = 8y
```

**Parameters**: k₁ = 1.0, k₂ = 4.0, k₃ = 1.2

**Key Properties**:
- Lyapunov exponents: λ₁ > 0, λ₂ > 0 (hyperchaotic)
- Sensitive dependence: |δZ(t)| ≈ |δZ₀|e^(λt)
- Ergodicity ensures uniform distribution

### 2. Hopfield Neural Network

The Hopfield network stores and retrieves cryptographic patterns:

**Energy Function**:
```
E = -½ ∑ᵢ∑ⱼ wᵢⱼsᵢsⱼ + ∑ᵢ θᵢsᵢ
```

**Update Rule**:
```
sᵢ(t+1) = sign(∑ⱼ wᵢⱼsⱼ(t) - θᵢ)
```

**Weight Matrix (Hebbian Learning)**:
```
wᵢⱼ = (1/N) ∑ᵏ ξᵢᵏξⱼᵏ  (i ≠ j)
wᵢᵢ = 0
```

Where:
- N = number of patterns
- ξᵏ = k-th stored pattern
- sᵢ = neuron state (-1 or +1)
- θᵢ = threshold for neuron i

### 3. Shark Smell Optimization (SSO)

Bio-inspired optimization algorithm mimicking shark foraging behavior:

**Position Update**:
```
xᵢ(t+1) = xᵢ(t) + vᵢ(t+1)
```

**Velocity Update**:
```
vᵢ(t+1) = w·vᵢ(t) + c₁·r₁·(pbestᵢ - xᵢ(t)) + c₂·r₂·(gbest - xᵢ(t)) + c₃·r₃·(smell_gradient)
```

**Smell Function**:
```
smell(x) = 1/(1 + distance_to_prey) · concentration_factor
```

**Parameters**:
- w = inertia weight (0.4 - 0.9)
- c₁, c₂, c₃ = acceleration coefficients
- r₁, r₂, r₃ = random values [0,1]

### 4. Integrated Cryptographic Scheme

#### Key Generation Process:
1. **Chaotic Key Stream**: Generate initial keys using hyperchaotic system
2. **Hopfield Validation**: Store key patterns in Hopfield network for validation
3. **SSO Optimization**: Optimize chaotic parameters for maximum entropy

#### Authentication Protocol:
1. **Pattern Storage**: Store user authentication patterns in Hopfield network
2. **Chaotic Challenge**: Generate challenge using chaotic system
3. **Pattern Matching**: Verify response using Hopfield pattern retrieval
4. **SSO Adaptation**: Adapt system parameters based on attack patterns

#### Encryption Scheme:
```
C = P ⊕ K_chaos ⊕ H_pattern
```

Where:
- C = ciphertext
- P = plaintext  
- K_chaos = chaotic keystream
- H_pattern = Hopfield-generated mask

## Project Architecture

```
chaos-hopfield-sso/
├── README.md
├── requirements.txt
├── main.py
├── config/
│   ├── __init__.py
│   └── system_config.py
├── chaos/
│   ├── __init__.py
│   ├── chebyshev.py
│   ├── hyperchaos.py
│   └── key_generator.py
├── neural/
│   ├── __init__.py
│   ├── hopfield.py
│   ├── pattern_storage.py
│   └── authenticator.py
├── sso/
│   ├── __init__.py
│   ├── optimizer.py
│   └── parameter_tuner.py
├── crypto/
│   ├── __init__.py
│   ├── cipher.py
│   ├── key_manager.py
│   └── protocol.py
├── api/
│   ├── __init__.py
│   ├── chaos_api.py
│   ├── neural_api.py
│   ├── sso_api.py
│   └── crypto_api.py
├── utils/
│   ├── __init__.py
│   ├── math_utils.py
│   └── validation.py
└── tests/
    ├── __init__.py
    ├── test_chaos.py
    ├── test_hopfield.py
    ├── test_sso.py
    └── test_integration.py
```

## API Structure

### Chaos API
```python
class ChaosAPI:
    def generate_sequence(initial_state, length)
    def generate_keystream(seed, length)
    def get_lyapunov_exponents()
```

### Hopfield API
```python
class HopfieldAPI:
    def store_patterns(patterns)
    def retrieve_pattern(input_pattern)
    def validate_key(key_pattern)
```

### SSO API
```python
class SSOAPI:
    def optimize_parameters(objective_function, bounds)
    def adapt_system(performance_metrics)
    def get_optimal_config()
```

### Crypto API
```python
class CryptoAPI:
    def encrypt(plaintext, chaos_key, hopfield_mask)
    def decrypt(ciphertext, chaos_key, hopfield_mask)
    def authenticate(challenge, response)
```

## Key Features

1. **Multi-layer Security**: Combines chaos, neural networks, and optimization
2. **Adaptive Defense**: SSO adapts parameters against attacks
3. **Pattern-based Auth**: Hopfield networks provide robust authentication
4. **High Entropy**: Chaotic systems ensure cryptographic randomness
5. **Modular Design**: Clean API separation for each component

## Security Properties

- **Confusion**: Chaotic nonlinearity obscures plaintext-ciphertext relationship
- **Diffusion**: Hopfield networks spread influence across entire key space
- **Adaptability**: SSO provides dynamic parameter adjustment
- **Key Sensitivity**: Small key changes cause dramatic output changes
- **Statistical Security**: Combined systems pass randomness tests

## Performance Considerations

- **Chaos Generation**: O(n) for sequence of length n
- **Hopfield Retrieval**: O(N²) where N is pattern size
- **SSO Optimization**: O(P×G) where P is population, G is generations
- **Overall Complexity**: Suitable for real-time applications with proper parameter tuning

## Dependencies

- numpy: Numerical computations
- scipy: Scientific computing and optimization
- matplotlib: Visualization (optional)
- pytest: Testing framework

## Usage Example

```python
from api import ChaosAPI, HopfieldAPI, SSOAPI, CryptoAPI

# Initialize systems
chaos = ChaosAPI()
hopfield = HopfieldAPI()
sso = SSOAPI()
crypto = CryptoAPI()

# Generate optimized parameters
optimal_params = sso.optimize_parameters(chaos.entropy_objective, bounds)

# Generate chaos-based key
chaos_key = chaos.generate_keystream(seed, length)

# Create Hopfield authentication pattern
auth_pattern = hopfield.store_patterns([user_biometric])

# Encrypt data
ciphertext = crypto.encrypt(plaintext, chaos_key, auth_pattern)
```

This framework provides a novel approach to cryptography by leveraging the complementary strengths of chaotic dynamics, neural pattern recognition, and bio-inspired optimization.