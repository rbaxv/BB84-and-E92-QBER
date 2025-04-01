import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time
from scipy.stats import norm

class QuantumProtocolSimulator:
    """Base class for quantum key distribution protocols."""
    
    def __init__(self, num_qubits=1000, error_rate=0.0, eavesdropping=False):
        self.num_qubits = num_qubits
        self.error_rate = error_rate
        self.eavesdropping = eavesdropping
        self.qber_history = []
        self.current_step = 0
        self.steps_history = []
        
    def generate_random_bits(self, n):
        """Generate n random bits (0 or 1)."""
        return np.random.randint(0, 2, n)
    
    def apply_channel_noise(self, bit):
        """Apply random noise to the bit based on error_rate."""
        if random.random() < self.error_rate:
            return 1 - bit  # Flip the bit
        return bit
    
    def estimate_security(self):
        """
        Estimate if the key exchange is secure based on QBER.
        """
        if self.qber > 0.11:
            return False, "QBER too high, likely eavesdropping"
        elif self.qber > 0.06:
            return False, "QBER concerning, possible interference"
        else:
            return True, "QBER acceptable"
            
    def sacrifice_key_portion(self, percentage=0.2):
        """
        Sacrifice a portion of the key to check for eavesdropping.
        """
        if self.sifted_key_alice is None or len(self.sifted_key_alice) == 0:
            return 0
            
        # Choose random indices to sacrifice
        num_to_sacrifice = int(len(self.sifted_key_alice) * percentage)
        sacrifice_indices = np.random.choice(
            len(self.sifted_key_alice), 
            size=num_to_sacrifice, 
            replace=False
        )
        
        # Calculate QBER on sacrificed bits
        alice_sacrificed = self.sifted_key_alice[sacrifice_indices]
        bob_sacrificed = self.sifted_key_bob[sacrifice_indices]
        errors = np.sum(alice_sacrificed != bob_sacrificed)
        sacrificed_qber = errors / len(alice_sacrificed) if len(alice_sacrificed) > 0 else 0
        
        # Remove sacrificed bits from the key
        mask = np.ones(len(self.sifted_key_alice), dtype=bool)
        mask[sacrifice_indices] = False
        self.sifted_key_alice = self.sifted_key_alice[mask]
        self.sifted_key_bob = self.sifted_key_bob[mask]
        
        return sacrificed_qber
    
    def privacy_amplification(self):
        """
        Perform a simple privacy amplification by applying a hash function.
        """
        if len(self.sifted_key_alice) < 2:
            return 0
            
        # Combining consecutive pairs of bits with XOR
        final_key_length = len(self.sifted_key_alice) // 2
        alice_final_key = np.zeros(final_key_length, dtype=int)
        bob_final_key = np.zeros(final_key_length, dtype=int)
        
        for i in range(final_key_length):
            alice_final_key[i] = self.sifted_key_alice[2*i] ^ self.sifted_key_alice[2*i+1]
            bob_final_key[i] = self.sifted_key_bob[2*i] ^ self.sifted_key_bob[2*i+1]
        
        self.sifted_key_alice = alice_final_key
        self.sifted_key_bob = bob_final_key
        
        return final_key_length
    
    def get_protocol_summary(self):
        """Get a summary of the protocol execution."""
        if self.sifted_key_alice is None:
            return "Protocol not yet executed."
            
        is_secure, reason = self.estimate_security()
        
        summary = {
            "qubits_sent": self.num_qubits,
            "matching_bases": len(self.matching_bases_indices) if hasattr(self, 'matching_bases_indices') else 'N/A',
            "sifted_key_length": len(self.sifted_key_alice),
            "errors": np.sum(self.sifted_key_alice != self.sifted_key_bob),
            "qber": self.qber,
            "secure": is_secure,
            "security_assessment": reason,
            "protocol": self.__class__.__name__
        }
        
        return summary


class BB84Simulator(QuantumProtocolSimulator):
    """
    Simulator for the BB84 Quantum Key Distribution protocol with QBER calculation.
    """
    
    # Constants for bases and bit values
    RECTILINEAR_BASIS = 0  # + basis (horizontal/vertical polarization)
    DIAGONAL_BASIS = 1     # × basis (diagonal polarization)
    
    def __init__(self, num_qubits=1000, error_rate=0.0, eavesdropping=False):
        """
        Initialize the BB84 simulator.
        """
        super().__init__(num_qubits, error_rate, eavesdropping)
        
        # Protocol components
        self.alice_bits = None
        self.alice_bases = None
        self.bob_bases = None
        self.bob_measurements = None
        self.matching_bases_indices = None
        self.sifted_key_alice = None
        self.sifted_key_bob = None
        self.qber = None
    
    def generate_random_bases(self, n):
        """Generate n random bases (0: rectilinear, 1: diagonal)."""
        return np.random.randint(0, 2, n)
    
    def measure_qubit(self, bit, prepared_basis, measuring_basis):
        """
        Simulate the measurement of a qubit.
        """
        if prepared_basis == measuring_basis:
            # If bases match, result is deterministic (same as prepared bit)
            return bit
        else:
            # If bases don't match, result is random
            return random.randint(0, 1)
    
    def run_protocol(self):
        """Execute the full BB84 protocol."""
        # Step 1: Alice generates random bits and bases
        self.alice_bits = self.generate_random_bits(self.num_qubits)
        self.alice_bases = self.generate_random_bases(self.num_qubits)
        
        # Step 2: Bob generates random bases for measurement
        self.bob_bases = self.generate_random_bases(self.num_qubits)
        
        # Step 3: Bob measures qubits
        self.bob_measurements = np.zeros(self.num_qubits, dtype=int)
        
        # Simulate the quantum transmission
        for i in range(self.num_qubits):
            measured_bit = self.alice_bits[i]
            
            # Simulate Eve's intervention if eavesdropping is enabled
            if self.eavesdropping:
                eve_basis = self.generate_random_bases(1)[0]
                measured_bit = self.measure_qubit(measured_bit, self.alice_bases[i], eve_basis)
                measured_bit = self.measure_qubit(measured_bit, eve_basis, self.bob_bases[i])
            else:
                # Regular transmission (Alice to Bob)
                measured_bit = self.measure_qubit(measured_bit, self.alice_bases[i], self.bob_bases[i])
            
            # Apply channel noise
            measured_bit = self.apply_channel_noise(measured_bit)
            
            self.bob_measurements[i] = measured_bit
        
        # Step 4: Basis reconciliation - find where Alice and Bob used the same basis
        self.matching_bases_indices = np.where(self.alice_bases == self.bob_bases)[0]
        
        # Step 5: Create sifted keys
        self.sifted_key_alice = self.alice_bits[self.matching_bases_indices]
        self.sifted_key_bob = self.bob_measurements[self.matching_bases_indices]
        
        # Step 6: Calculate QBER
        errors = np.sum(self.sifted_key_alice != self.sifted_key_bob)
        self.qber = errors / len(self.sifted_key_alice) if len(self.sifted_key_alice) > 0 else 0
        self.qber_history.append(self.qber)
        
    def step_simulation(self):
        """Execute the protocol step by step for visualization."""
        self.current_step += 1
        
        if self.current_step == 1:
            # Alice generates bits and bases
            self.alice_bits = self.generate_random_bits(self.num_qubits)
            self.alice_bases = self.generate_random_bases(self.num_qubits)
            return "Alice generated random bits and bases"
            
        elif self.current_step == 2:
            # Bob generates bases
            self.bob_bases = self.generate_random_bases(self.num_qubits)
            return "Bob generated random bases for measurement"
            
        elif self.current_step == 3:
            # Quantum transmission
            self.bob_measurements = np.zeros(self.num_qubits, dtype=int)
            for i in range(self.num_qubits):
                measured_bit = self.alice_bits[i]
                
                if self.eavesdropping:
                    eve_basis = self.generate_random_bases(1)[0]
                    measured_bit = self.measure_qubit(measured_bit, self.alice_bases[i], eve_basis)
                    measured_bit = self.measure_qubit(measured_bit, eve_basis, self.bob_bases[i])
                else:
                    measured_bit = self.measure_qubit(measured_bit, self.alice_bases[i], self.bob_bases[i])
                
                measured_bit = self.apply_channel_noise(measured_bit)
                self.bob_measurements[i] = measured_bit
                
            return "Bob measured the received qubits"
            
        elif self.current_step == 4:
            # Basis reconciliation
            self.matching_bases_indices = np.where(self.alice_bases == self.bob_bases)[0]
            return f"Alice and Bob found {len(self.matching_bases_indices)} matching bases"
            
        elif self.current_step == 5:
            # Create sifted keys
            self.sifted_key_alice = self.alice_bits[self.matching_bases_indices]
            self.sifted_key_bob = self.bob_measurements[self.matching_bases_indices]
            return f"Sifted key created with length {len(self.sifted_key_alice)}"
            
        elif self.current_step == 6:
            # Calculate QBER
            errors = np.sum(self.sifted_key_alice != self.sifted_key_bob)
            self.qber = errors / len(self.sifted_key_alice) if len(self.sifted_key_alice) > 0 else 0
            self.qber_history.append(self.qber)
            self.steps_history.append(self.current_step)
            return f"QBER calculated: {self.qber:.4f}"
            
        elif self.current_step == 7:
            # Security assessment
            is_secure, reason = self.estimate_security()
            return f"Security assessment: {'Secure' if is_secure else 'Not secure'} - {reason}"
            
        elif self.current_step == 8:
            # Reset for next round
            self.current_step = 0
            return "Simulation complete, reset for next round"
            
        return "Unknown step"


class E91Simulator(QuantumProtocolSimulator):
    """
    Simulator for the E91 (Ekert91) Quantum Key Distribution protocol with QBER calculation.
    Based on quantum entanglement and Bell's inequality testing.
    """
    
    # Constants for measurement angles (optimized for maximum Bell violation)
    # Alice's angles: 0, π/4, π/2
    # Bob's angles: π/4, 3π/4, π/2
    # These angles are chosen to maximize Bell inequality violation
    ALICE_ANGLES = [0, np.pi/4, np.pi/2]
    BOB_ANGLES = [np.pi/4, 3*np.pi/4, np.pi/2]
    
    def __init__(self, num_qubits=1000, error_rate=0.0, eavesdropping=False):
        """Initialize the E91 simulator."""
        super().__init__(num_qubits, error_rate, eavesdropping)
        self.bell_history = []
        self.alice_angle_choices = None
        self.bob_angle_choices = None
        self.alice_measurements = None
        self.bob_measurements = None
        self.matching_angles_indices = None
        self.sifted_key_alice = None
        self.sifted_key_bob = None
        self.qber = None
        self.bell_parameter = None
        
    def quantum_correlation(self, angle_diff):
        """
        Calculate quantum correlation for the singlet state.
        For a singlet state |ψ⁻⟩ = (|01⟩ - |10⟩)/√2, correlation is -cos(θ)
        where θ is the angle between measurement directions.
        """
        return -np.cos(angle_diff)
        
    def measure_entangled_pair(self, alice_angle, bob_angle):
        """
        Measure an entangled pair in the singlet state.
        Returns a correlated pair of measurements (alice_result, bob_result).
        """
        # Calculate ideal quantum correlation
        angle_diff = alice_angle - bob_angle
        correlation = self.quantum_correlation(angle_diff)
        
        # Perfect anti-correlation at same angle (θ = 0)
        if abs(angle_diff) < 1e-10:  # Numerical precision check
            alice_result = random.randint(0, 1)
            bob_result = 1 - alice_result
        else:
            # Probability of same result is (1 - correlation)/2
            prob_same = (1 - correlation) / 2
            alice_result = random.randint(0, 1)
            bob_result = alice_result if random.random() > prob_same else 1 - alice_result
            
        # Apply noise independently to each particle
        if random.random() < self.error_rate:
            alice_result = 1 - alice_result
        if random.random() < self.error_rate:
            bob_result = 1 - bob_result
            
        return alice_result, bob_result
        
    def calculate_bell_parameter(self):
        """
        Calculate Bell's parameter S using CHSH inequality.
        For maximally entangled singlet state, expect S = 2√2 ≈ 2.828 in ideal case.
        """
        if not hasattr(self, 'alice_measurements') or len(self.alice_measurements) == 0:
            return 0
            
        # Initialize correlation arrays
        correlations = np.zeros((3, 3))  # 3x3 matrix for all angle combinations
        counts = np.zeros((3, 3))
        
        # Calculate correlations for each angle combination
        for i in range(self.num_qubits):
            a_idx = self.alice_angle_choices[i]
            b_idx = self.bob_angle_choices[i]
            
            # Convert to ±1 basis
            alice_result = 2 * self.alice_measurements[i] - 1
            bob_result = 2 * self.bob_measurements[i] - 1
            
            correlations[a_idx, b_idx] += alice_result * bob_result
            counts[a_idx, b_idx] += 1
            
        # Normalize correlations
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.divide(correlations, counts, where=counts != 0)
            correlations = np.nan_to_num(correlations)
        
        # Calculate CHSH S parameter using optimal angle combinations
        # S = E(a₁,b₁) - E(a₁,b₂) + E(a₂,b₁) + E(a₂,b₂)
        # Using angles that maximize violation
        S = correlations[0,0] - correlations[0,1] + correlations[1,0] + correlations[1,1]
        
        return abs(S)
        
    def run_protocol(self):
        """Execute the full E91 protocol."""
        # Step 1: Alice and Bob choose random measurement angles
        self.alice_angle_choices = np.random.randint(0, len(self.ALICE_ANGLES), self.num_qubits)
        self.bob_angle_choices = np.random.randint(0, len(self.BOB_ANGLES), self.num_qubits)
        
        # Step 2: Generate and measure entangled pairs
        self.alice_measurements = np.zeros(self.num_qubits, dtype=int)
        self.bob_measurements = np.zeros(self.num_qubits, dtype=int)
        
        for i in range(self.num_qubits):
            # Get measurement angles
            alice_angle = self.ALICE_ANGLES[self.alice_angle_choices[i]]
            bob_angle = self.BOB_ANGLES[self.bob_angle_choices[i]]
            
            if self.eavesdropping:
                # Eve's measurement breaks entanglement
                self.alice_measurements[i] = random.randint(0, 1)
                self.bob_measurements[i] = random.randint(0, 1)
            else:
                # Perform quantum measurement
                alice_result, bob_result = self.measure_entangled_pair(alice_angle, bob_angle)
                self.alice_measurements[i] = alice_result
                self.bob_measurements[i] = bob_result
        
        # Step 3: Find matching angles for key generation
        # In E91, we use anti-parallel measurements for key generation
        self.matching_angles_indices = []
        for i in range(self.num_qubits):
            if ((self.alice_angle_choices[i] == 0 and self.bob_angle_choices[i] == 0) or  # 0° vs π/4
                (self.alice_angle_choices[i] == 2 and self.bob_angle_choices[i] == 2)):   # π/2 vs π/2
                self.matching_angles_indices.append(i)
        
        # Step 4: Create sifted keys
        self.matching_angles_indices = np.array(self.matching_angles_indices)
        if len(self.matching_angles_indices) > 0:
            self.sifted_key_alice = self.alice_measurements[self.matching_angles_indices]
            self.sifted_key_bob = self.bob_measurements[self.matching_angles_indices]
            # For anti-parallel measurements, Bob needs to flip his bits
            self.sifted_key_bob = 1 - self.sifted_key_bob
        else:
            self.sifted_key_alice = np.array([], dtype=int)
            self.sifted_key_bob = np.array([], dtype=int)
        
        # Step 5: Calculate QBER
        if len(self.sifted_key_alice) > 0:
            errors = np.sum(self.sifted_key_alice != self.sifted_key_bob)
            self.qber = errors / len(self.sifted_key_alice)
        else:
            self.qber = 0
            
        self.qber_history.append(self.qber)
        
        # Step 6: Calculate and store Bell parameter
        self.bell_parameter = self.calculate_bell_parameter()
        self.bell_history.append(self.bell_parameter)
        
    def step_simulation(self):
        """Execute the protocol step by step for visualization."""
        self.current_step += 1
        
        if self.current_step == 1:
            # Alice and Bob choose measurement angles
            self.alice_angle_choices = np.random.randint(0, len(self.ALICE_ANGLES), self.num_qubits)
            self.bob_angle_choices = np.random.randint(0, len(self.BOB_ANGLES), self.num_qubits)
            return "Alice and Bob chose random measurement angles"
            
        elif self.current_step == 2:
            # Generate entangled pairs and measure
            self.alice_measurements = np.zeros(self.num_qubits, dtype=int)
            self.bob_measurements = np.zeros(self.num_qubits, dtype=int)
            
            for i in range(self.num_qubits):
                entangled_bit = self.generate_entangled_pair()
                alice_angle = self.ALICE_ANGLES[self.alice_angle_choices[i]]
                bob_angle = self.BOB_ANGLES[self.bob_angle_choices[i]]
                
                alice_result = self.measure_particle(0, alice_angle, entangled_bit)
                bob_result = self.measure_particle(0, bob_angle, entangled_bit)
                
                if self.eavesdropping:
                    alice_result = random.randint(0, 1)
                    bob_result = random.randint(0, 1)
                
                alice_result = self.apply_channel_noise(alice_result)
                bob_result = self.apply_channel_noise(bob_result)
                
                self.alice_measurements[i] = alice_result
                self.bob_measurements[i] = bob_result
                
            return "Generated and measured entangled pairs"
            
        elif self.current_step == 3:
            # Find matching angles
            self.matching_angles_indices = []
            for i in range(self.num_qubits):
                if ((self.alice_angle_choices[i] == 0 and self.bob_angle_choices[i] == 0) or
                    (self.alice_angle_choices[i] == 2 and self.bob_angle_choices[i] == 1)):
                    self.matching_angles_indices.append(i)
                    
            self.matching_angles_indices = np.array(self.matching_angles_indices)
            return f"Found {len(self.matching_angles_indices)} matching angles for key generation"
            
        elif self.current_step == 4:
            # Create sifted keys
            if len(self.matching_angles_indices) > 0:
                self.sifted_key_alice = self.alice_measurements[self.matching_angles_indices]
                self.sifted_key_bob = self.bob_measurements[self.matching_angles_indices]
                self.sifted_key_bob = 1 - self.sifted_key_bob
            else:
                self.sifted_key_alice = np.array([], dtype=int)
                self.sifted_key_bob = np.array([], dtype=int)
                
            return f"Sifted key created with length {len(self.sifted_key_alice)}"
            
        elif self.current_step == 5:
            # Calculate QBER
            if len(self.sifted_key_alice) > 0:
                errors = np.sum(self.sifted_key_alice != self.sifted_key_bob)
                self.qber = errors / len(self.sifted_key_alice)
            else:
                self.qber = 0
                
            self.qber_history.append(self.qber)
            return f"QBER calculated: {self.qber:.4f}"
            
        elif self.current_step == 6:
            # Calculate Bell parameter
            self.bell_parameter = self.calculate_bell_parameter()
            return f"Bell parameter calculated: {self.bell_parameter:.4f}"
            
        elif self.current_step == 7:
            # Security assessment
            is_secure, reason = self.estimate_security()
            bell_security = abs(self.bell_parameter) > 2
            
            bell_msg = f"Bell test {'passed' if bell_security else 'failed'}"
            return f"Security assessment: {'Secure' if is_secure else 'Not secure'} - {reason}, {bell_msg}"
            
        elif self.current_step == 8:
            # Reset for next round
            self.current_step = 0
            return "Simulation complete, reset for next round"
            
        return "Unknown step"

    def get_protocol_summary(self):
        """Get a summary of the E91 protocol execution."""
        if self.sifted_key_alice is None:
            return "Protocol not yet executed."
            
        is_secure, reason = self.estimate_security()
        bell_security = abs(self.bell_parameter) > 2 if self.bell_parameter is not None else False
        
        summary = {
            "qubits_sent": self.num_qubits,
            "matching_angles": len(self.matching_angles_indices) if hasattr(self, 'matching_angles_indices') else 0,
            "sifted_key_length": len(self.sifted_key_alice) if self.sifted_key_alice is not None else 0,
            "errors": np.sum(self.sifted_key_alice != self.sifted_key_bob) if self.sifted_key_alice is not None else 0,
            "qber": self.qber if hasattr(self, 'qber') else 0,
            "bell_parameter": self.bell_parameter if hasattr(self, 'bell_parameter') else None,
            "bell_test": "Passed" if bell_security else "Failed",
            "secure": is_secure and bell_security,
            "security_assessment": reason,
            "protocol": "E91"
        }
        
        return summary


class QKDSimulatorGUI:
    """GUI for the Quantum Key Distribution protocol simulator."""
    
    def __init__(self, master=None):
        """Initialize the GUI."""
        self.bb84_simulator = BB84Simulator(num_qubits=1000, error_rate=0.01)
        self.e91_simulator = E91Simulator(num_qubits=1000, error_rate=0.01)
        self.current_simulator = self.bb84_simulator  # Default to BB84
        
        self.eavesdropping_enabled = False
        self.noise_level = 0.01
        self.continuous_run = False
        self.protocol = "BB84"  # Start with BB84
        self.fig = None
        self.last_update_time = time.time()
        self.update_interval = 0.1  # Update every 100ms
        self.setup_plots()
        
    def setup_plots(self):
        """Setup matplotlib plots."""
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 9))
        self.fig.suptitle('Quantum Key Distribution Protocol Simulator', fontsize=16)
        
        # QBER Monitor
        self.qber_line_bb84, = self.axs[0, 0].plot([], [], 'r-', linewidth=2, label='BB84 QBER')
        self.qber_line_e91, = self.axs[0, 0].plot([], [], 'b-', linewidth=2, label='E91 QBER')
        self.axs[0, 0].set_xlim(0, 20)
        self.axs[0, 0].set_ylim(0, 0.3)
        self.axs[0, 0].set_title('QBER Monitor')
        self.axs[0, 0].set_xlabel('Iteration')
        self.axs[0, 0].set_ylabel('QBER')
        self.axs[0, 0].grid(True)
        self.axs[0, 0].axhline(y=0.11, color='r', linestyle='--', label='Critical QBER')
        self.axs[0, 0].axhline(y=0.06, color='y', linestyle='--', label='Warning QBER')
        self.axs[0, 0].legend()
        
        # Bits comparison
        self.axs[0, 1].set_title('Alice vs Bob Bits (Sample)')
        self.axs[0, 1].set_xlabel('Bit Position')
        self.axs[0, 1].set_ylabel('Bit Value')
        self.axs[0, 1].grid(True)
        
        # Protocol Statistics
        self.stats_text = self.axs[1, 0].text(0.05, 0.5, 'Run simulation to see statistics', 
                                              transform=self.axs[1, 0].transAxes, fontsize=10)
        self.axs[1, 0].set_title('Protocol Statistics')
        self.axs[1, 0].axis('off')
        
        # Bell Parameter (for E91) or Key Rate vs QBER (for BB84)
        self.axs[1, 1].set_title('Protocol Specific Metrics')
        self.axs[1, 1].grid(True)
        
        # Initial setup for BB84's key rate plot
        self.setup_key_rate_plot()
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Add controls
        plt.subplots_adjust(bottom=0.2)
        
        # Add buttons
        self.axbutton_run = plt.axes([0.1, 0.05, 0.12, 0.05])
        self.button_run = plt.Button(self.axbutton_run, 'Run Simulation')
        self.button_run.on_clicked(self.run_simulation)
        
        self.axbutton_toggle_eve = plt.axes([0.25, 0.05, 0.12, 0.05])
        self.button_toggle_eve = plt.Button(self.axbutton_toggle_eve, 'Toggle Eve: OFF')
        self.button_toggle_eve.on_clicked(self.toggle_eavesdropping)
        
        self.axbutton_noise = plt.axes([0.4, 0.05, 0.12, 0.05])
        self.button_noise = plt.Button(self.axbutton_noise, f'Noise: {self.noise_level:.2f}')
        self.button_noise.on_clicked(self.change_noise)
        
        self.axbutton_continuous = plt.axes([0.6, 0.05, 0.12, 0.05])
        self.button_continuous = plt.Button(self.axbutton_continuous, 'Continuous: OFF')
        self.button_continuous.on_clicked(self.toggle_continuous)
        
        # Add protocol toggle button
        self.axbutton_protocol = plt.axes([0.75, 0.05, 0.12, 0.05])
        self.button_protocol = plt.Button(self.axbutton_protocol, f'Protocol: {self.protocol}')
        self.button_protocol.on_clicked(self.toggle_protocol)
        
    def setup_key_rate_plot(self):
        """Setup the key rate plot for BB84."""
        self.axs[1, 1].clear()
        self.axs[1, 1].set_title('Key Rate vs QBER')
        self.axs[1, 1].set_xlabel('QBER')
        self.axs[1, 1].set_ylabel('Key Rate')
        # Create a theoretical curve
        qber_range = np.linspace(0, 0.15, 100)
        key_rate = np.maximum(1 - 2 * self.h_binary(qber_range), 0)
        self.axs[1, 1].plot(qber_range, key_rate, 'b-', label='Theoretical Rate')
        self.axs[1, 1].scatter([], [], color='r', s=50, label='Current Rate')
        self.axs[1, 1].grid(True)
        self.axs[1, 1].legend()
        
    def h_binary(self, x):
        """Binary entropy function."""
        result = np.zeros_like(x)
        mask = (x > 0) & (x < 1)
        x_masked = x[mask]
        result[mask] = -x_masked * np.log2(x_masked) - (1 - x_masked) * np.log2(1 - x_masked)
        return result
        
    def update_plots(self, i):
        """Update plots with current simulation data."""
        current_time = time.time()
        
        # Only run new iteration if in continuous mode and enough time has passed
        if self.continuous_run and (current_time - self.last_update_time) >= self.update_interval:
            self.run_one_iteration()
            self.last_update_time = current_time
            
        # Update QBER monitor
        qber_history = self.current_simulator.qber_history
        if qber_history:
            x = list(range(1, len(qber_history) + 1))
            if self.protocol == "BB84":
                self.qber_line_bb84.set_data(x, qber_history)
            else:
                self.qber_line_e91.set_data(x, qber_history)
            
            if len(qber_history) > 20:
                self.axs[0, 0].set_xlim(len(qber_history) - 20, len(qber_history))
                
        # Update bits comparison
        self.axs[0, 1].clear()
        self.axs[0, 1].set_title('Alice vs Bob Bits (Sample)')
        self.axs[0, 1].set_xlabel('Bit Position')
        self.axs[0, 1].set_ylabel('Bit Value')
        self.axs[0, 1].grid(True)
        
        if self.current_simulator.sifted_key_alice is not None and len(self.current_simulator.sifted_key_alice) > 0:
            # Show a sample of the first 20 bits or fewer
            sample_size = min(20, len(self.current_simulator.sifted_key_alice))
            x = np.arange(sample_size)
            
            alice_sample = self.current_simulator.sifted_key_alice[:sample_size]
            bob_sample = self.current_simulator.sifted_key_bob[:sample_size]
            
            # Plot Alice's bits
            self.axs[0, 1].stem(x, alice_sample, linefmt='b-', markerfmt='bo', label='Alice')
            
            # Plot Bob's bits
            self.axs[0, 1].stem(x, bob_sample, linefmt='g-', markerfmt='g^', label='Bob')
            
            # Highlight errors
            errors = alice_sample != bob_sample
            if np.any(errors):
                error_indices = x[errors]
                self.axs[0, 1].plot(error_indices, bob_sample[errors], 'rx', markersize=10, label='Errors')
                
            self.axs[0, 1].set_ylim(-0.1, 1.1)
            self.axs[0, 1].set_yticks([0, 1])
            self.axs[0, 1].legend()
            
        # Update protocol statistics
        if self.current_simulator.sifted_key_alice is not None:
            summary = self.current_simulator.get_protocol_summary()
            stats_str = f"Protocol: {self.protocol}\n"
            stats_str += f"Qubits Sent: {summary['qubits_sent']}\n"
            
            if self.protocol == "BB84":
                stats_str += f"Matching Bases: {summary.get('matching_bases', 'N/A')} "
                if 'matching_bases' in summary:
                    stats_str += f"({summary['matching_bases']/summary['qubits_sent']*100:.1f}%)\n"
            else:  # E91
                stats_str += f"Matching Angles: {summary.get('matching_angles', 'N/A')} "
                if 'matching_angles' in summary:
                    stats_str += f"({summary['matching_angles']/summary['qubits_sent']*100:.1f}%)\n"
                
            stats_str += f"Sifted Key Length: {summary['sifted_key_length']}\n"
            stats_str += f"Errors: {summary['errors']}\n"
            stats_str += f"QBER: {summary['qber']:.4f}\n"
            
            if self.protocol == "E91":
                stats_str += f"Bell Parameter: {summary.get('bell_parameter', 'N/A'):.4f}\n"
                stats_str += f"Bell Test: {summary.get('bell_test', 'N/A')}\n"
                
            stats_str += f"Security: {summary['security_assessment']}"
            
            self.stats_text.set_text(stats_str)
            
            # Update protocol specific plot
            if self.protocol == "BB84":
                if len(qber_history) > 0:
                    key_rate = summary['sifted_key_length'] / summary['qubits_sent'] * (1 - 2 * self.h_binary(np.array([summary['qber']]))[0])
                    key_rate = max(0, key_rate)
                    # Update scatter plot point
                    if len(self.axs[1, 1].get_lines()) > 1:  # Check if plot exists
                        self.axs[1, 1].get_lines()[-1].set_data([summary['qber']], [key_rate])
            else:  # E91
                if len(qber_history) > 0 and hasattr(self.current_simulator, 'bell_history'):
                    x = list(range(1, len(self.current_simulator.bell_history) + 1))
                    self.axs[1, 1].clear()
                    self.axs[1, 1].set_title('Bell Parameter')
                    self.axs[1, 1].set_xlabel('Iteration')
                    self.axs[1, 1].set_ylabel('Bell Parameter')
                    self.axs[1, 1].axhline(y=2, color='r', linestyle='--', label='Classical Limit')
                    self.axs[1, 1].plot([1, max(20, len(x))], [2.828, 2.828], 'b-', label='Quantum Limit (2√2)')
                    self.axs[1, 1].plot(x, self.current_simulator.bell_history, 'g-', label='Measured')
                    self.axs[1, 1].set_ylim(1.8, 3.0)
                    self.axs[1, 1].grid(True)
                    self.axs[1, 1].legend()
                    
        return [self.qber_line_bb84, self.qber_line_e91]
        
    def run_one_iteration(self):
        """Run one complete iteration of the current protocol."""
        # Configure simulator
        self.current_simulator.error_rate = self.noise_level
        self.current_simulator.eavesdropping = self.eavesdropping_enabled
        
        # Run protocol
        self.current_simulator.run_protocol()
        
    def run_simulation(self, event):
        """Run the BB84 simulation."""
        self.run_one_iteration()
        self.update_plots(0)
        plt.draw()
        
    def toggle_eavesdropping(self, event):
        """Toggle eavesdropping simulation."""
        self.eavesdropping_enabled = not self.eavesdropping_enabled
        self.button_toggle_eve.label.set_text(f"Toggle Eve: {'ON' if self.eavesdropping_enabled else 'OFF'}")
        plt.draw()
        
    def change_noise(self, event):
        """Change the noise level."""
        self.noise_level = (self.noise_level + 0.01) % 0.12
        self.button_noise.label.set_text(f'Noise: {self.noise_level:.2f}')
        plt.draw()
        
    def toggle_continuous(self, event):
        """Toggle continuous running mode."""
        self.continuous_run = not self.continuous_run
        self.button_continuous.label.set_text(f"Continuous: {'ON' if self.continuous_run else 'OFF'}")
        plt.draw()
        
    def toggle_protocol(self, event):
        """Toggle between BB84 and E91 protocols."""
        if self.protocol == "BB84":
            self.protocol = "E91"
            self.current_simulator = self.e91_simulator
            # Clear BB84 specific plots but keep QBER history
            self.axs[1, 1].clear()
            self.axs[1, 1].set_title('Bell Parameter')
            self.axs[1, 1].set_xlabel('Iteration')
            self.axs[1, 1].set_ylabel('Bell Parameter')
            self.axs[1, 1].axhline(y=2, color='r', linestyle='--', label='Classical Limit')
            self.axs[1, 1].plot([], [], 'b-', label='Quantum Limit (2√2 ≈ 2.828)')
            self.axs[1, 1].set_ylim(1.8, 3.0)
            self.axs[1, 1].grid(True)
            self.axs[1, 1].legend()
        else:
            self.protocol = "BB84"
            self.current_simulator = self.bb84_simulator
            # Restore BB84 key rate plot but keep QBER history
            self.axs[1, 1].clear()
            self.setup_key_rate_plot()
            
        self.button_protocol.label.set_text(f'Protocol: {self.protocol}')
        # Don't reset QBER history when switching protocols
        plt.draw()
        
    def run(self):
        """Start the GUI."""
        ani = FuncAnimation(self.fig, self.update_plots, frames=None, interval=100, blit=True)
        plt.show()

# Entry point for the application
def main():
    """Main function to run the QBER ANALYSIS GUI."""
    print("Starting  Quantum Key Distribution Simulator")
    print("This simulator demonstrates the BB84 & E91 protocol and calculates QBER")
    
    gui = QKDSimulatorGUI()
    gui.run()

if __name__ == "__main__":
    main()