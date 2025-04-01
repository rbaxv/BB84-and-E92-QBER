# BB84-and-E92-QBER
# Quantum Key Distribution Simulator

A Python-based simulator for quantum key distribution protocols including BB84 and E91 (Ekert91) with real-time visualization and analysis of Quantum Bit Error Rate (QBER).

## Features

- Simulation of BB84 and E91 quantum key distribution protocols
- Real-time monitoring of Quantum Bit Error Rate (QBER)
- Interactive visualization of key generation process
- Eavesdropping detection and simulation
- Bell's inequality testing for E91 protocol
- Noise simulation and analysis

## Installation

### Prerequisites

```bash
# Required Python packages
pip install numpy matplotlib scipy
```

### Running the Simulator

```bash
python quantum_key_distribution_simulator.py
```

## Usage

The GUI provides several interactive controls:
- Toggle between BB84 and E91 protocols
- Enable/disable eavesdropper simulation
- Adjust noise levels
- Run single or continuous simulations

## Project Structure

- `QBER.py`: Main application file

## Dependencies

- Python 3.9+
- NumPy
- Matplotlib
- SciPy

## License

[MIT](https://choosealicense.com/licenses/mit/)
