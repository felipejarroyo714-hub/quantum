"""
========================================================================
EXPANDED FAULT-TOLERANT QUANTUM SIMULATOR WITH MULTIPLE BACKENDS
========================================================================

This version integrates Qiskit Aer and Cirq as backends through PennyLane plugins
to support simulations with more than 23 qubits.
"""

import os
import time
import json
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Quantum computing libraries
try:
    import pennylane as qml
    from pennylane import numpy as qnp
except ImportError:
    print("Installing required packages...")
    os.system("pip install pennylane -q")
    import pennylane as qml
    from pennylane import numpy as qnp

# High precision arithmetic
try:
    import mpmath as mp
    mp.mp.dps = 50  # Set decimal places of precision
except ImportError:
    print("Installing mpmath...")
    os.system("pip install mpmath -q")
    import mpmath as mp
    mp.mp.dps = 50

# Check and install backend plugins
def install_and_check_backends():
    """Install and check available backend plugins."""
    backends = {}
    
    # Check for Qiskit plugin
    try:
        import pennylane_qiskit
        backends['qiskit'] = True
        print("PennyLane-Qiskit plugin is available")
    except ImportError:
        print("Installing PennyLane-Qiskit plugin...")
        os.system("pip install pennylane-qiskit -q")
        try:
            import pennylane_qiskit
            backends['qiskit'] = True
            print("PennyLane-Qiskit plugin installed successfully")
        except ImportError:
            print("Failed to install PennyLane-Qiskit plugin")
            backends['qiskit'] = False
    
    # Check for Cirq plugin
    try:
        import pennylane_cirq
        backends['cirq'] = True
        print("PennyLane-Cirq plugin is available")
    except ImportError:
        print("Installing PennyLane-Cirq plugin...")
        os.system("pip install pennylane-cirq -q")
        try:
            import pennylane_cirq
            backends['cirq'] = True
            print("PennyLane-Cirq plugin installed successfully")
        except ImportError:
            print("Failed to install PennyLane-Cirq plugin")
            backends['cirq'] = False
    
    return backends

# Install and check backends
AVAILABLE_BACKENDS = install_and_check_backends()

# Check device capabilities for each backend
def check_backend_capabilities(backend_name):
    """Check the maximum number of qubits supported by a backend."""
    try:
        if backend_name == "qiskit" and AVAILABLE_BACKENDS.get('qiskit', False):
            # Test Qiskit Aer simulator
            from qiskit import Aer
            simulator = Aer.get_backend('aer_simulator_statevector')
            
            # Test with increasing number of qubits
            max_qubits = 30  # Start with a reasonable number
            for n in range(5, 31):
                try:
                    from qiskit import QuantumCircuit
                    qc = QuantumCircuit(n)
                    qc.h(0)
                    for i in range(n-1):
                        qc.cx(i, i+1)
                    
                    # Try to run the circuit
                    from qiskit import execute
                    job = execute(qc, simulator, shots=1)
                    result = job.result()
                    max_qubits = n
                except:
                    break
            
            print(f"Qiskit Aer simulator supports up to {max_qubits} qubits")
            return max_qubits, "qiskit.aer_simulator"
        
        elif backend_name == "cirq" and AVAILABLE_BACKENDS.get('cirq', False):
            # Test Cirq simulator
            import cirq
            
            # Test with increasing number of qubits
            max_qubits = 30  # Start with a reasonable number
            for n in range(5, 31):
                try:
                    qubits = cirq.LineQubit.range(n)
                    circuit = cirq.Circuit()
                    circuit.append(cirq.H(qubits[0]))
                    for i in range(n-1):
                        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
                    
                    # Try to simulate
                    simulator = cirq.Simulator()
                    result = simulator.simulate(circuit)
                    max_qubits = n
                except:
                    break
            
            print(f"Cirq simulator supports up to {max_qubits} qubits")
            return max_qubits, "cirq.simulator"
        
        else:
            # Default PennyLane device
            max_qubits = 23  # Known limit for default.mixed
            print(f"Default PennyLane device supports up to {max_qubits} qubits")
            return max_qubits, "default.mixed"
    
    except Exception as e:
        print(f"Error checking {backend_name} capabilities: {e}")
        return 10, "default.qubit"  # Fallback

# Get capabilities for all backends
BACKEND_CAPABILITIES = {}
for backend in ['default', 'qiskit', 'cirq']:
    if backend == 'default' or AVAILABLE_BACKENDS.get(backend, False):
        max_qubits, device_name = check_backend_capabilities(backend)
        BACKEND_CAPABILITIES[backend] = {
            'max_qubits': max_qubits,
            'device_name': device_name
        }

# Select the best backend
BEST_BACKEND = max(BACKEND_CAPABILITIES.keys(), 
                  key=lambda k: BACKEND_CAPABILITIES[k]['max_qubits'])
MAX_SUPPORTED_QUBITS = BACKEND_CAPABILITIES[BEST_BACKEND]['max_qubits']
DEVICE_NAME = BACKEND_CAPABILITIES[BEST_BACKEND]['device_name']

print(f"Using {BEST_BACKEND} backend with support for {MAX_SUPPORTED_QUBITS} qubits")

# Mathematical constants with high precision
class MathematicalConstants:
    """High-precision mathematical constants used throughout the simulator."""
    
    def __init__(self, precision=50):
        mp.mp.dps = precision
        self.phi = (mp.mpf(1) + mp.sqrt(5)) / 2  # Golden ratio
        self.sqrt5 = mp.sqrt(5)
        self.sqrt6 = mp.sqrt(6)
        self.sqrt6_over_2 = self.sqrt6 / 2
        self.pi = mp.pi
        self.e = mp.e
        self.alpha = self.sqrt6_over_2 / self.phi  # Scaling equivalence constant
        
    def transform_to_tetrahedral_basis(self, value):
        """Transform a value to the tetrahedral basis."""
        return value * self.sqrt6_over_2
    
    def transform_to_golden_basis(self, value):
        """Transform a value to the golden ratio basis."""
        return value * self.phi
    
    def blend_bases(self, value, blend_factor):
        """Blend between tetrahedral and golden bases."""
        tetrahedral = self.transform_to_tetrahedral_basis(value)
        golden = self.transform_to_golden_basis(value)
        return blend_factor * tetrahedral + (1 - blend_factor) * golden

# High precision number handling
class HPNumber:
    """High-precision number wrapper for mpmath."""
    
    def __init__(self, value, precision=50):
        mp.mp.dps = precision
        if isinstance(value, str):
            self.value = mp.mpf(value)
        elif isinstance(value, (int, float)):
            self.value = mp.mpf(str(value))
        else:
            self.value = mp.mpf(value)
    
    def __add__(self, other):
        return HPNumber(self.value + HPNumber(other).value)
    
    def __sub__(self, other):
        return HPNumber(self.value - HPNumber(other).value)
    
    def __mul__(self, other):
        return HPNumber(self.value * HPNumber(other).value)
    
    def __truediv__(self, other):
        return HPNumber(self.value / HPNumber(other).value)
    
    def __pow__(self, other):
        return HPNumber(self.value ** HPNumber(other).value)
    
    def __eq__(self, other):
        return self.value == HPNumber(other).value
    
    def __repr__(self):
        return str(self.value)
    
    def __str__(self):
        return str(self.value)
    
    def to_float(self):
        """Convert to standard float (may lose precision)."""
        return float(self.value)
    
    def to_complex(self):
        """Convert to complex number."""
        return complex(self.value)

# Precision handler for converting between standard and high-precision numbers
class PrecisionHandler:
    """Handles conversion between standard and high-precision numbers."""
    
    def __init__(self, precision=50):
        self.precision = precision
        self.constants = MathematicalConstants(precision)
    
    def to_hp_array(self, array):
        """Convert numpy array to high-precision array."""
        return np.array([HPNumber(x, self.precision) for x in array])
    
    def from_hp_array(self, hp_array):
        """Convert high-precision array back to numpy array."""
        return np.array([x.to_float() for x in hp_array])
    
    def to_complex_hp_array(self, array):
        """Convert numpy complex array to high-precision complex array."""
        result = []
        for x in array:
            if np.iscomplex(x):
                real = HPNumber(x.real, self.precision)
                imag = HPNumber(x.imag, self.precision)
                result.append(real + imag * HPNumber(1j, self.precision))
            else:
                result.append(HPNumber(x, self.precision))
        return np.array(result)
    
    def from_complex_hp_array(self, hp_array):
        """Convert high-precision complex array back to numpy array."""
        result = []
        for x in hp_array:
            if isinstance(x.value, mp.mpc):
                result.append(complex(x.value))
            else:
                result.append(float(x.value))
        return np.array(result)

# Enhanced configuration system with backend selection
@dataclass
class FaultTolerantSimulationConfig:
    """
    Centralized configuration for the fault-tolerant quantum simulation framework.
    """
    # --- Simulation Metadata ---
    simulation_name: str = "fault_tolerant_qec_simulation"
    description: str = "Multi-backend simulation of fault-tolerant quantum computation."
    
    # --- Quantum Hardware & Backend ---
    backend: str = BEST_BACKEND  # Use the best available backend
    quantum_device: str = DEVICE_NAME  # Use the best available device
    n_qubits: int = min(25, MAX_SUPPORTED_QUBITS)  # Stay within device limits
    shots: int = 1024  # Number of measurement shots
    
    # --- Noise Model Parameters ---
    noise_params: Optional[Dict[str, float]] = None
    
    # --- Core Quantum Circuit Configuration ---
    base_precision: int = 30  # Precision for high-accuracy calculations
    entanglement_plugin: str = "vortex_knot"  # Entanglement pattern for the circuit
    
    # --- Fault-Tolerance & QEC Options ---
    enable_magic_state_distillation: bool = False  # Disabled by default
    magic_distillation_rounds: int = 3  # Reduced rounds
    enable_majorana_simulation: bool = False  # Disabled by default
    
    # --- Surface Code QEC Specifics ---
    qec_code: str = "repetition_code"  # Use simpler code for memory
    code_distance: int = 3  # Small distance for memory efficiency
    
    # --- Execution & Output ---
    verbose_logging: bool = True
    output_directory: str = "./fault_tolerant_outputs"
    
    # --- Advanced Features ---
    enable_hamiltonian_evolution: bool = False  # Disabled for memory
    enable_adaptive_time_stepping: bool = False  # Disabled for memory
    enable_cognitive_decoder: bool = False  # Disabled for memory
    enable_algebraic_compression: bool = False  # Disabled for memory
    
    # --- Physical Parameters ---
    t1_time: float = 100.0  # Relaxation time in microseconds
    t2_time: float = 50.0   # Dephasing time in microseconds
    error_correction_threshold: float = 0.05  # Threshold for QEC
    
    # --- Majorana Parameters ---
    majorana_braiding_steps: int = 5  # Reduced steps
    
    def __post_init__(self):
        if self.noise_params is None:
            self.noise_params = {'bit_flip_prob': 0.01, 'phase_flip_prob': 0.005}
        
        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Adjust qubit count based on device and code
        if self.qec_code == "surface_code_2d":
            # Surface code needs more qubits, adjust if necessary
            required_qubits = self.code_distance * self.code_distance
            if required_qubits > MAX_SUPPORTED_QUBITS:
                # Reduce code distance
                self.code_distance = int(np.sqrt(MAX_SUPPORTED_QUBITS)) - 1
                if self.code_distance % 2 == 0:
                    self.code_distance -= 1
                self.n_qubits = self.code_distance * self.code_distance
                print(f"Adjusted code distance to {self.code_distance} for device limitations")
        else:
            # Ensure we don't exceed device limits
            self.n_qubits = min(self.n_qubits, MAX_SUPPORTED_QUBITS)

# Main configuration to be used
CONFIG = FaultTolerantSimulationConfig(
    noise_params={'bit_flip_prob': 0.01, 'phase_flip_prob': 0.005}
)

# ==================== ENHANCED CORE COMPONENTS ====================

# Backend Factory for creating devices
class BackendFactory:
    """Factory for creating quantum devices from different backends."""
    
    @staticmethod
    def create_device(config: FaultTolerantSimulationConfig):
        """Create a quantum device based on the backend configuration."""
        if config.backend == "qiskit" and AVAILABLE_BACKENDS.get('qiskit', False):
            # Use Qiskit backend
            if config.noise_params:
                # Use noisy simulator
                device = qml.device('qiskit.aer', wires=config.n_qubits, 
                                   shots=config.shots, backend='aer_simulator_density_matrix')
            else:
                # Use state vector simulator
                device = qml.device('qiskit.aer', wires=config.n_qubits, 
                                   shots=config.shots, backend='aer_simulator_statevector')
        
        elif config.backend == "cirq" and AVAILABLE_BACKENDS.get('cirq', False):
            # Use Cirq backend
            device = qml.device('cirq.simulator', wires=config.n_qubits, shots=config.shots)
        
        else:
            # Use default PennyLane backend
            if config.noise_params:
                device = qml.device('default.mixed', wires=config.n_qubits, shots=config.shots)
            else:
                device = qml.device('default.qubit', wires=config.n_qubits, shots=config.shots)
        
        return device

# Quantum Memory API with backend awareness
class QuantumMemoryAPI:
    """Handles saving and loading simulation states with backend awareness."""
    
    def __init__(self, config: FaultTolerantSimulationConfig):
        self.config = config
        self.precision_handler = PrecisionHandler(config.base_precision)
        self.constants = MathematicalConstants(config.base_precision)
        
    def save_simulation_state(self, state_data: Dict[str, Any], filename: Optional[str] = None):
        """Save simulation state to disk with backend information."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"{self.config.simulation_name}_{self.config.backend}_{timestamp}.pkl"
        
        filepath = os.path.join(self.config.output_directory, filename)
        
        # Add backend information to state data
        state_data['backend'] = self.config.backend
        state_data['device'] = self.config.quantum_device
        
        # Compress state data if needed
        if 'state_vector' in state_data:
            state_data = self._compress_state(state_data)
        
        # Save with protocol 4 for compatibility
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f, protocol=4)
        
        if self.config.verbose_logging:
            print(f"Simulation state saved to {filepath}")
        
        return filepath
    
    def load_simulation_state(self, filepath: str):
        """Load simulation state from disk."""
        with open(filepath, 'rb') as f:
            state_data = pickle.load(f)
        
        # Decompress if needed
        if 'compressed' in state_data and state_data['compressed']:
            state_data = self._decompress_state(state_data)
        
        if self.config.verbose_logging:
            backend = state_data.get('backend', 'unknown')
            device = state_data.get('device', 'unknown')
            print(f"Simulation state loaded from {filepath} (backend: {backend}, device: {device})")
        
        return state_data
    
    def _compress_state(self, state_data: Dict[str, Any]):
        """Compress state data for memory efficiency."""
        state_vector = state_data.get('state_vector')
        if state_vector is None:
            return state_data
        
        # Convert to float32 to save memory
        if state_vector.dtype == np.complex128:
            state_vector = state_vector.astype(np.complex64)
        elif state_vector.dtype == np.float64:
            state_vector = state_vector.astype(np.float32)
        
        # Update state data
        new_state_data = state_data.copy()
        new_state_data['state_vector'] = state_vector
        new_state_data['compressed'] = True
        
        return new_state_data
    
    def _decompress_state(self, state_data: Dict[str, Any]):
        """Decompress state data."""
        if 'state_vector' not in state_data or not state_data.get('compressed', False):
            return state_data
        
        # Convert back to original dtype if needed
        state_vector = state_data['state_vector']
        if state_vector.dtype == np.complex64:
            state_vector = state_vector.astype(np.complex128)
        elif state_vector.dtype == np.float32:
            state_vector = state_vector.astype(np.float64)
        
        # Update state data
        new_state_data = state_data.copy()
        new_state_data['state_vector'] = state_vector
        return new_state_data

# Repetition Code for memory efficiency
class RepetitionCode:
    """Memory-efficient repetition code implementation."""
    
    def __init__(self, distance: int):
        self.distance = distance
        self.n_qubits = distance
        self.logical_0 = np.zeros(2**distance, dtype=complex)
        self.logical_0[0] = 1  # |000...0>
        self.logical_1 = np.zeros(2**distance, dtype=complex)
        self.logical_1[-1] = 1  # |111...1>
    
    def stabilizer_observables(self):
        """Return stabilizer observables for measurement."""
        observables = []
        
        # Z stabilizers between neighboring qubits
        for i in range(self.distance - 1):
            pauli_string = ['I'] * self.distance
            pauli_string[i] = 'Z'
            pauli_string[i+1] = 'Z'
            observables.append(''.join(pauli_string))
        
        return observables
    
    def logical_pauli_word(self, which: str):
        """Return logical Pauli operators."""
        if which == 'X':
            # Logical X is X on all qubits
            return 'X' * self.distance
        elif which == 'Z':
            # Logical Z is Z on first qubit
            pauli_string = ['I'] * self.distance
            pauli_string[0] = 'Z'
            return ''.join(pauli_string)
        else:
            raise ValueError("which must be 'X' or 'Z'")
    
    def decode_corrections(self, syndrome_bits):
        """Simple decoding for repetition code."""
        corrections = {"X": [], "Z": []}
        
        # Count number of 1s in syndrome
        n_ones = sum(syndrome_bits)
        
        # If odd number of 1s, flip the last qubit
        if n_ones % 2 == 1:
            corrections["X"].append(self.distance - 1)
        
        return corrections

# Surface Code with backend awareness
class SurfaceCode2D:
    """Memory-efficient implementation of the 2D surface code."""
    
    def __init__(self, distance: int):
        # Check for valid distance
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Code distance must be an odd integer >= 3")
        
        self.distance = distance
        self.lattice_size = distance * distance
        
        # Limit size for memory efficiency
        if self.lattice_size > MAX_SUPPORTED_QUBITS:
            raise ValueError(f"Surface code with distance {distance} requires {self.lattice_size} qubits, "
                           f"but device only supports {MAX_SUPPORTED_QUBITS}")
        
        # Simplified stabilizers
        self.n_stabilizers = (distance - 1) * (distance - 1)
        
        # Define logical operators
        self.logical_x = self._define_logical_X()
        self.logical_z = self._define_logical_Z()
    
    def _define_logical_X(self):
        """Define logical X operator."""
        # Simplified: X on first row
        return [(0, j) for j in range(self.distance)]
    
    def _define_logical_Z(self):
        """Define logical Z operator."""
        # Simplified: Z on first column
        return [(i, 0) for i in range(self.distance)]
    
    def stabilizer_observables(self):
        """Return stabilizer observables for measurement."""
        observables = []
        
        # Simplified: only use Z stabilizers
        for i in range(0, self.distance - 1, 2):
            for j in range(0, self.distance - 1, 2):
                # Create plaquette stabilizer
                pauli_string = ['I'] * self.lattice_size
                
                # Add Z operators to plaquette
                positions = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
                for pi, pj in positions:
                    if pi < self.distance and pj < self.distance:
                        idx = pi * self.distance + pj
                        pauli_string[idx] = 'Z'
                
                observables.append(''.join(pauli_string))
        
        return observables[:self.n_stabilizers]  # Limit number of stabilizers
    
    def logical_pauli_word(self, which: str):
        """Return logical Pauli operators."""
        if which == 'X':
            pauli_string = ['I'] * self.lattice_size
            for i, j in self.logical_x:
                idx = i * self.distance + j
                pauli_string[idx] = 'X'
            return ''.join(pauli_string)
        elif which == 'Z':
            pauli_string = ['I'] * self.lattice_size
            for i, j in self.logical_z:
                idx = i * self.distance + j
                pauli_string[idx] = 'Z'
            return ''.join(pauli_string)
        else:
            raise ValueError("which must be 'X' or 'Z'")
    
    def decode_corrections(self, x_syndrome_bits, z_syndrome_bits, cognitive_weights=None):
        """Simplified decoding for memory efficiency."""
        corrections = {"X": [], "Z": []}
        
        # Simple majority vote for Z corrections
        if z_syndrome_bits and sum(z_syndrome_bits) > len(z_syndrome_bits) / 2:
            # Apply Z correction to center qubit
            center = self.distance // 2
            corrections["Z"].append(center * self.distance + center)
        
        return corrections

# Backend-aware QEC layer
class IntegratedQEC:
    """Backend-aware integrated quantum error correction layer."""
    
    def __init__(self, config: FaultTolerantSimulationConfig):
        self.config = config
        
        # Initialize the appropriate QEC code
        if config.qec_code == "surface_code_2d":
            try:
                self.code = SurfaceCode2D(distance=config.code_distance)
            except ValueError as e:
                print(f"Cannot use surface code: {e}")
                print("Falling back to repetition code")
                self.code = RepetitionCode(distance=min(config.code_distance, config.n_qubits))
                self.config.qec_code = "repetition_code"
        elif config.qec_code == "repetition_code":
            self.code = RepetitionCode(distance=min(config.code_distance, config.n_qubits))
        else:
            print(f"Unknown QEC code: {config.qec_code}, using repetition code")
            self.code = RepetitionCode(distance=min(config.code_distance, config.n_qubits))
            self.config.qec_code = "repetition_code"
        
        # Initialize measurement device using backend factory
        self.device = BackendFactory.create_device(config)
    
    def run_qec_cycle(self, current_state):
        """Run a QEC cycle with backend awareness."""
        # Get stabilizer observables
        stabilizers = self.code.stabilizer_observables()
        
        # Measure stabilizers
        syndrome_bits = []
        
        for stabilizer in stabilizers:
            # Convert stabilizer string to observable
            obs = self._pauli_string_to_observable(stabilizer)
            
            # Measure stabilizer
            if len(current_state.shape) == 1:  # Pure state
                # Calculate expectation value
                exp_val = np.real(np.vdot(current_state, obs @ current_state))
            else:  # Density matrix
                exp_val = np.real(np.trace(current_state @ obs))
            
            # Convert to syndrome bit
            syndrome_bit = 0 if exp_val > 0 else 1
            syndrome_bits.append(syndrome_bit)
        
        # Decode syndromes to get corrections
        if self.config.qec_code == "repetition_code":
            corrections = self.code.decode_corrections(syndrome_bits)
        else:
            # Split syndrome bits for surface code
            n_x = len(self.code.x_stabilizers) if hasattr(self.code, 'x_stabilizers') else 0
            x_syndrome_bits = syndrome_bits[:n_x]
            z_syndrome_bits = syndrome_bits[n_x:]
            corrections = self.code.decode_corrections(x_syndrome_bits, z_syndrome_bits)
        
        return corrections
    
    def _pauli_string_to_observable(self, pauli_string):
        """Convert a Pauli string to an observable matrix."""
        n_qubits = len(pauli_string)
        dim = 2**n_qubits
        
        # For memory efficiency, use sparse representation for large systems
        if dim > 2**10:  # More than 10 qubits
            return self._pauli_string_to_observable_sparse(pauli_string)
        
        obs = np.eye(1, dtype=complex)
        
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        # Build observable
        for pauli in pauli_string:
            obs = np.kron(obs, pauli_map[pauli])
        
        return obs
    
    def _pauli_string_to_observable_sparse(self, pauli_string):
        """Convert Pauli string to sparse observable for memory efficiency."""
        # Simplified: return diagonal matrix for Z-only stabilizers
        n_qubits = len(pauli_string)
        dim = 2**n_qubits
        
        obs = np.eye(dim, dtype=complex)
        
        # Apply Z operations
        for i, pauli in enumerate(pauli_string):
            if pauli == 'Z':
                # Apply Z to qubit i
                for j in range(dim):
                    if (j >> i) & 1:  # If qubit i is 1
                        obs[j, j] *= -1
        
        return obs

# Backend-aware quantum circuit engine
class SuperExponentialRadialTraversalCircuit:
    """Backend-aware quantum circuit engine."""
    
    def __init__(self, config: FaultTolerantSimulationConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        
        # Create device using backend factory
        self.device = BackendFactory.create_device(config)
        
        # Initialize arithmetic module with reduced precision
        self.arithmetic = QuantumArithmeticModule(config.base_precision)
    
    def build_circuit(self, logical_operations: List[Callable]) -> qml.QNode:
        """Build the quantum circuit with logical operations."""
        
        @qml.qnode(self.device)
        def circuit():
            # State initialization
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.PauliZ(wires=i)
            
            # Apply logical operations
            for operation in logical_operations:
                operation(circuit, list(range(self.n_qubits)))
                
                # Apply noise after each operation if noise is enabled
                if self.config.noise_params:
                    self._apply_noise(circuit, list(range(self.n_qubits)))
            
            # Return state
            return qml.state()
        
        return circuit
    
    def _apply_noise(self, circuit, qubits):
        """Apply noise channels to qubits."""
        noise_params = self.config.noise_params
        
        # Apply bit-flip noise
        if 'bit_flip_prob' in noise_params:
            for qubit in qubits:
                qml.BitFlip(noise_params['bit_flip_prob'], wires=qubit)
        
        # Apply phase-flip noise
        if 'phase_flip_prob' in noise_params:
            for qubit in qubits:
                qml.PhaseFlip(noise_params['phase_flip_prob'], wires=qubit)

# Simplified quantum arithmetic
class QuantumArithmeticModule:
    """Simplified quantum arithmetic for memory efficiency."""
    
    def __init__(self, precision=30):
        self.precision = precision
        mp.mp.dps = precision
    
    def calculate_angle(self, value):
        """Calculate a rotation angle."""
        return float(np.arcsin(np.clip(value, -1, 1)))
    
    def calculate_phase(self, value):
        """Calculate a phase."""
        return float(np.angle(complex(value, 0)))

# Backend-aware simulator
class FaultTolerantSimulator:
    """Backend-aware fault-tolerant quantum computation simulator."""
    
    def __init__(self, config: FaultTolerantSimulationConfig):
        self.config = config
        
        # Initialize components
        self.core_circuit_engine = SuperExponentialRadialTraversalCircuit(config)
        self.qec_layer = IntegratedQEC(config)
        self.memory_api = QuantumMemoryAPI(config)
    
    def run_simulation_workflow(self, logical_operations: List[Callable]):
        """Run the complete simulation workflow."""
        if self.config.verbose_logging:
            print(f"Starting simulation: {self.config.simulation_name}")
            print(f"Using {self.config.n_qubits} qubits with {self.config.qec_code}")
            print(f"Backend: {self.config.backend}, Device: {self.config.quantum_device}")
        
        # Step 1: Generate ideal state
        if self.config.verbose_logging:
            print("Generating ideal state...")
        
        # Temporarily disable noise
        original_noise_params = self.config.noise_params
        self.config.noise_params = None
        
        # Build and execute circuit without noise
        ideal_circuit = self.core_circuit_engine.build_circuit(logical_operations)
        psi_ideal = ideal_circuit()
        
        # Restore noise parameters
        self.config.noise_params = original_noise_params
        
        # Step 2: Run noisy, corrected execution
        if self.config.verbose_logging:
            print("Running noisy, corrected execution...")
        
        # Initialize state
        current_state = psi_ideal.copy()
        
        # Track history
        syndrome_history = []
        correction_history = []
        fidelity_history = []
        
        # Execute with error correction
        current_state = self._standard_execution(logical_operations, current_state, 
                                               syndrome_history, correction_history, fidelity_history)
        
        # Step 3: Analysis and output
        if self.config.verbose_logging:
            print("Analyzing results...")
        
        # Calculate final fidelity
        final_fidelity = self._calculate_fidelity(current_state, psi_ideal)
        
        # Calculate logical expectation values
        logical_x_exp = self._calculate_logical_expectation(current_state, 'X')
        logical_z_exp = self._calculate_logical_expectation(current_state, 'Z')
        
        # Prepare results
        results = {
            'config': self.config,
            'ideal_state': psi_ideal,
            'final_state': current_state,
            'final_fidelity': final_fidelity,
            'logical_x_expectation': logical_x_exp,
            'logical_z_expectation': logical_z_exp,
            'syndrome_history': syndrome_history,
            'correction_history': correction_history,
            'fidelity_history': fidelity_history
        }
        
        # Step 4: Persistence
        if self.config.verbose_logging:
            print("Saving results...")
        
        filepath = self.memory_api.save_simulation_state(results)
        
        if self.config.verbose_logging:
            print(f"Simulation complete. Final fidelity: {final_fidelity:.4f}")
            print(f"Results saved to: {filepath}")
        
        return results
    
    def _standard_execution(self, logical_operations, initial_state, syndrome_history, correction_history, fidelity_history):
        """Execute with standard QEC cycles."""
        current_state = initial_state
        
        # For each logical operation
        for i, operation in enumerate(logical_operations):
            if self.config.verbose_logging:
                print(f"Applying logical operation {i+1}/{len(logical_operations)}")
            
            # Apply operation
            circuit = self.core_circuit_engine.build_circuit([operation])
            current_state = circuit()
            
            # Run QEC cycle
            corrections = self.qec_layer.run_qec_cycle(current_state)
            syndrome_history.append(corrections)
            correction_history.append(corrections)
            
            # Apply corrections
            current_state = self._apply_corrections(current_state, corrections)
            
            # Calculate fidelity
            fidelity = self._calculate_fidelity(current_state, initial_state)
            fidelity_history.append(fidelity)
            
            if self.config.verbose_logging:
                print(f"  Fidelity after QEC: {fidelity:.4f}")
        
        return current_state
    
    def _apply_corrections(self, state, corrections):
        """Apply correction operations to a quantum state."""
        if len(state.shape) == 1:  # Pure state
            # Convert to density matrix for easier manipulation
            rho = np.outer(state, np.conj(state))
        else:  # Already a density matrix
            rho = state
        
        # Apply X corrections
        for wire in corrections.get('X', []):
            rho = self._apply_pauli(rho, wire, 'X')
        
        # Apply Z corrections
        for wire in corrections.get('Z', []):
            rho = self._apply_pauli(rho, wire, 'Z')
        
        # Return as same type as input
        if len(state.shape) == 1:  # Return pure state
            # Extract pure state from density matrix (simplified)
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
            max_idx = np.argmax(eigenvalues)
            return eigenvectors[:, max_idx]
        else:  # Return density matrix
            return rho
    
    def _apply_pauli(self, rho, wire, pauli_type):
        """Apply a Pauli operator to a density matrix."""
        n_qubits = int(np.log2(len(rho)))
        dim = 2**n_qubits
        
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        pauli_map = {'X': X, 'Y': Y, 'Z': Z}
        
        # Create operator for the specified wire
        op = I
        for i in range(n_qubits):
            if i == wire:
                op = np.kron(op, pauli_map[pauli_type])
            else:
                op = np.kron(op, I)
        
        # Apply operator
        return op @ rho @ np.conj(op.T)
    
    def _calculate_fidelity(self, state1, state2):
        """Calculate fidelity between two quantum states."""
        if len(state1.shape) == 1 and len(state2.shape) == 1:  # Both pure states
            overlap = np.abs(np.vdot(state1, state2))**2
            return overlap
        elif len(state1.shape) == 2 and len(state2.shape) == 1:  # Mixed vs pure
            fidelity = np.real(np.vdot(state2, state1 @ state2))
            return fidelity
        elif len(state1.shape) == 1 and len(state2.shape) == 2:  # Pure vs mixed
            fidelity = np.real(np.vdot(state1, state2 @ state1))
            return fidelity
        else:  # Both mixed states
            return np.real(np.trace(state1 @ state2))**2
    
    def _calculate_logical_expectation(self, state, logical_op):
        """Calculate expectation value of a logical operator."""
        # Get logical operator from code
        pauli_string = self.qec_layer.code.logical_pauli_word(logical_op)
        
        # Convert to observable matrix
        obs = self.qec_layer._pauli_string_to_observable(pauli_string)
        
        # Calculate expectation value
        if len(state.shape) == 1:  # Pure state
            exp_val = np.real(np.vdot(state, obs @ state))
        else:  # Density matrix
            exp_val = np.real(np.trace(state @ obs))
        
        return exp_val

# Example logical operations (memory-efficient)
def hadamard_on_all(circuit, qubits):
    """Apply Hadamard to all qubits."""
    for qubit in qubits:
        circuit(qml.Hadamard(wires=qubit))

def cnot_chain(circuit, qubits):
    """Apply CNOT in a chain pattern."""
    for i in range(len(qubits) - 1):
        circuit(qml.CNOT(wires=[qubits[i], qubits[i+1]]))

def create_bell_state(circuit, qubits):
    """Create a Bell state on the first two qubits."""
    if len(qubits) >= 2:
        circuit(qml.Hadamard(wires=qubits[0]))
        circuit(qml.CNOT(wires=[qubits[0], qubits[1]]))

def single_qubit_rotation(circuit, qubits):
    """Apply single qubit rotations."""
    if len(qubits) >= 1:
        circuit(qml.RX(np.pi/4, wires=qubits[0]))
        circuit(qml.RY(np.pi/4, wires=qubits[0]))

def quantum_fourier_transform(circuit, qubits):
    """Apply Quantum Fourier Transform."""
    n = len(qubits)
    for i in range(n):
        circuit(qml.Hadamard(wires=qubits[i]))
        for j in range(i+1, n):
            angle = np.pi / (2**(j-i))
            circuit(qml.RZ(angle, wires=qubits[j]))
            circuit(qml.CNOT(wires=[qubits[i], qubits[j]]))
            circuit(qml.RZ(-angle, wires=qubits[j]))
            circuit(qml.CNOT(wires=[qubits[i], qubits[j]]))

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-BACKEND FAULT-TOLERANT QUANTUM SIMULATOR")
    print("=" * 60)
    print(f"Available backends: {list(AVAILABLE_BACKENDS.keys())}")
    print(f"Best backend: {BEST_BACKEND} with {MAX_SUPPORTED_QUBITS} qubits")
    print()
    
    # Define logical operations (simplified for memory)
    logical_operations = [
        hadamard_on_all,
        cnot_chain,
        create_bell_state,
        quantum_fourier_transform
    ]
    
    # Create simulator with best backend
    simulator = FaultTolerantSimulator(CONFIG)
    
    # Run simulation
    results = simulator.run_simulation_workflow(logical_operations)
    
    # Print results
    print("\nSimulation Results:")
    print(f"Final Fidelity: {results['final_fidelity']:.4f}")
    print(f"Logical X Expectation: {results['logical_x_expectation']:.4f}")
    print(f"Logical Z Expectation: {results['logical_z_expectation']:.4f}")
    print(f"Number of QEC Cycles: {len(results['syndrome_history'])}")
    
    # Example of using different backends
    print("\n" + "=" * 60)
    print("COMPARING DIFFERENT BACKENDS")
    print("=" * 60)
    
    # Test with different backends if available
    for backend in ['default', 'qiskit', 'cirq']:
        if backend in AVAILABLE_BACKENDS and AVAILABLE_BACKENDS[backend]:
            max_qubits = BACKEND_CAPABILITIES[backend]['max_qubits']
            device_name = BACKEND_CAPABILITIES[backend]['device_name']
            
            print(f"\nTesting {backend} backend:")
            print(f"  Max qubits: {max_qubits}")
            print(f"  Device: {device_name}")
            
            # Create config for this backend
            backend_config = FaultTolerantSimulationConfig(
                backend=backend,
                quantum_device=device_name,
                n_qubits=min(10, max_qubits),  # Use smaller number for comparison
                simulation_name=f"backend_comparison_{backend}",
                noise_params={'bit_flip_prob': 0.01, 'phase_flip_prob': 0.005}
            )
            
            # Create and run simulator
            backend_simulator = FaultTolerantSimulator(backend_config)
            backend_results = backend_simulator.run_simulation_workflow([
                hadamard_on_all,
                cnot_chain,
                create_bell_state
            ])
            
            print(f"  Final Fidelity: {backend_results['final_fidelity']:.4f}")

