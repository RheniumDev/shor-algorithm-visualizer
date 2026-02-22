# Shor’s Algorithm Visualizer

## Overview
This project is an educational tool designed to demonstrate the mechanics of Shor’s algorithm for integer factorization. It provides a side-by-side comparison between classical modular exponentiation and quantum phase estimation (QPE). The application visualizes how quantum algorithms provide a theoretical exponential speedup by reducing the factoring problem to a period-finding problem in the modular group.

## Mathematical Workflow
The algorithm follows a rigorous mathematical process to factor a composite number N:
1. **Parameter Selection**: An integer 'a' is chosen such that 1 < a < N and gcd(a, N) = 1 (coprimality).
2. **Order Finding**: The algorithm seeks the smallest positive integer 'r', known as the order or period, such that a^r ≡ 1 (mod N).
3. **Period Extraction**: Quantum Phase Estimation is used to estimate the phase φ = s/r. Continued fraction reconstruction is then applied to derive the denominator 'r'.
4. **Factor Validation**: Once 'r' is obtained, it is validated for factoring suitability:
   - r must be greater than 0.
   - r must be even.
   - a^(r/2) must not be congruent to -1 (mod N).
5. **GCD Extraction**: If validated, non-trivial factors are calculated using gcd(a^(r/2) ± 1, N).

## Features
- **Interactive Visualization**: Real-time plotting of modular exponentiation functions and quantum measurement histograms.
- **Hybrid Computation Path**: Uses the Qiskit Aer 'aer_simulator' for full statevector simulation of small N (up to 32) and an idealized QPE emulator for larger values to maintain performance.
- **Automated Post-Processing**: Integrated continued fraction expansion for precise estimation of the order from noisy or probabilistic quantum output.
- **Robust Input Validation**: Strict checks for N value range, coprimality of 'a', and graceful handling of search failures.
- **Timeout Protection**: Internal monitors prevent the application from hanging during computationally intensive classical searches or complex circuit transpilation.

## Technical Implementation
The system architecture separates the UI, logic, and simulation layers:
- **Frontend**: Developed with Streamlit for a responsive, dashboard-style interface.
- **Quantum Simulation**: Implemented using the Qiskit framework. Circuits are constructed with controlled-U gate chains and an Inverse Quantum Fourier Transform (IQFT) for phase estimation.
- **Data Handling**: Uses NumPy and Pandas for numerical operations and managing measurement data.
- **Dynamic Optimization**: Computation parameters like qubit count (2·⌈log₂N⌉) and shot count are determined dynamically based on the input N to ensure statistical significance.

## Installation Steps
To run the visualizer locally:
1. Ensure Python 3.8+ is installed.
2. Install the required dependencies:
   ```bash
   pip install streamlit qiskit qiskit-aer matplotlib numpy pandas
   ```
3. Execute the application using the Streamlit CLI:
   ```bash
   streamlit run shor_app.py
   ```

## Complexity Discussion
- **Classical Scaling**: The most efficient known classical algorithm, the General Number Field Sieve (GNFS), operates in sub-exponential time. Naive period search scales linearly with N, which is exponential in the number of bits n.
- **Quantum Scaling**: Shor’s algorithm operates in polynomial time, specifically O(n²·log n·log log n) gates for a number with bit-length n.
- **Practical Considerations**: While Shor's algorithm points to a theoretical vulnerability in RSA cryptography, practical implementation for large N requires fault-tolerant quantum hardware with thousands of logical qubits, which is beyond the capacity of current Noisy Intermediate-Scale Quantum (NISQ) devices.

## Limitations
- **Simulation Constraints**: Full quantum simulation on classical hardware is memory-limited and scales exponentially with the number of qubits.
- **Idealized Noise**: The emulator uses Gaussian noise to simulate phase resolution but does not model specific hardware decoherence or gate errors.
- **Pedagogical Scope**: This tool is designed for educational demonstration and does not implement the specialized optimized circuits required for large-scale production factoring.

## Learning Outcomes
- Understanding the reduction of factoring to order-finding.
- Implementing controlled unitary operations and the Quantum Fourier Transform.
- Analyzing the role of continued fractions in quantum algorithms.
- Evaluating the performance gap between classical and quantum computational models.
- Practical experience with Streamlit UI development and Qiskit circuit simulation.
