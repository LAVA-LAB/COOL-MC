"""
Plot Quantum Circuit for Quantum REINFORCE Agent (Ski Configuration)

This script visualizes the quantum circuit used by the Quantum REINFORCE agent
for the ski environment. The circuit includes:
- Amplitude encoding for state preparation
- Variational layers with RX, RY, RZ rotations
- Entanglement layers with CNOT gates (ring topology)
- Measurements on all qubits
"""

import numpy as np
import pennylane as qml
import matplotlib
import matplotlib.pyplot as plt

# Configure matplotlib for PGF/LaTeX output with Unicode support
matplotlib.use('Agg')
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": False,  # Don't use LaTeX for rendering, only for PGF output
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}\usepackage{textcomp}\usepackage{amsmath,amssymb}",
})


def create_quantum_circuit_for_ski(num_qubits=4, num_layers=2):
    """
    Create the quantum circuit matching the ski configuration.

    Args:
        num_qubits: Number of qubits (default 4 for ski)
        num_layers: Number of variational layers (default 2 for ski)

    Returns:
        Quantum circuit function and device
    """
    # Create quantum device
    dev = qml.device('default.qubit', wires=num_qubits)

    # Number of parameters: 3 rotations per qubit per layer
    params_per_layer = num_qubits * 3
    total_params = params_per_layer * num_layers

    @qml.qnode(dev)
    def quantum_circuit(state_amplitudes, params):
        """
        Quantum circuit for policy network.

        Architecture:
        1. StatePrep: Amplitude encoding of classical state
        2. Variational layers: RX, RY, RZ rotations on each qubit
        3. Entanglement: CNOT gates in ring topology
        4. Measurements: PauliZ expectation values
        """
        # 1. Amplitude encoding - prepare quantum state
        qml.StatePrep(state_amplitudes, wires=range(num_qubits))

        # 2. Variational layers
        param_idx = 0
        for layer in range(num_layers):
            # Rotation layer - individual qubit rotations
            for qubit in range(num_qubits):
                qml.RX(params[param_idx], wires=qubit)
                qml.RY(params[param_idx + 1], wires=qubit)
                qml.RZ(params[param_idx + 2], wires=qubit)
                param_idx += 3

            # Entangling layer - skip on last layer
            if layer < num_layers - 1:
                for qubit in range(num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

                # Ring topology: connect last to first
                qml.CNOT(wires=[num_qubits - 1, 0])

        # 3. Measurements - Z expectation for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return quantum_circuit, dev, total_params


def plot_circuit(num_qubits=4, num_layers=2, save_path='quantum_circuit_ski.pgf'):
    """
    Create and plot the quantum circuit.

    Args:
        num_qubits: Number of qubits (default 4 for ski)
        num_layers: Number of variational layers (default 2 for ski)
        save_path: Path to save the figure
    """
    print(f"\n{'='*70}")
    print(f"Quantum REINFORCE Circuit for Ski Environment")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Number of qubits: {num_qubits}")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Parameters per layer: {num_qubits * 3} (3 rotations per qubit)")
    print(f"  - Total circuit parameters: {num_qubits * 3 * num_layers}")

    # Create circuit
    circuit, dev, total_params = create_quantum_circuit_for_ski(num_qubits, num_layers)

    # Create example state and parameters
    state_size = 2 ** num_qubits
    example_state = np.zeros(state_size)
    example_state[0] = 1.0  # Start in |0...0⟩ state

    # Random parameters (small values as in actual implementation)
    example_params = np.random.randn(total_params) * 0.01

    # Draw the circuit (decimals=None hides parameter values)
    fig, ax = qml.draw_mpl(circuit, decimals=None, style='pennylane')(example_state, example_params)

    # Adjust layout
    plt.tight_layout()

    # Save figure in both PGF and PNG formats
    pgf_path = save_path.replace('.png', '.pgf')
    png_path = save_path.replace('.pgf', '.png')

    # Save PNG first
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

    # Try to save PGF (may fail with Unicode characters from PennyLane)
    try:
        plt.savefig(pgf_path, format='pgf', bbox_inches='tight')
        print(f"\n✓ Circuit diagram saved to:")
        print(f"  - {pgf_path} (PGF for LaTeX)")
        print(f"  - {png_path} (PNG preview)")
    except Exception as e:
        # PGF export failed - save as PDF instead which can be included in LaTeX
        pdf_path = pgf_path.replace('.pgf', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"\n✓ Circuit diagram saved to:")
        print(f"  - {pdf_path} (PDF for LaTeX - use \\includegraphics)")
        print(f"  - {png_path} (PNG preview)")
        print(f"  Note: PGF export failed due to Unicode characters. PDF is a better option for LaTeX.")

    plt.close()

    print(f"\nCircuit Structure:")
    print(f"  1. StatePrep: Amplitude encoding of classical state")
    print(f"  2. Layer 0:")
    print(f"     - Rotations: RX, RY, RZ on each qubit (12 params)")
    print(f"     - Entanglement: CNOT gates in ring topology")
    print(f"  3. Layer 1:")
    print(f"     - Rotations: RX, RY, RZ on each qubit (12 params)")
    print(f"     - No entanglement (last layer)")
    print(f"  4. Measurements: PauliZ expectation values on all qubits")
    print(f"\nTotal trainable parameters in circuit: {total_params}")
    print(f"Plus classical head (linear layer): {num_qubits} × n_actions")
    print(f"{'='*70}\n")


def plot_circuit_with_noise(num_qubits=4, num_layers=2,
                            depolarizing_p=0.01,
                            save_path='quantum_circuit_ski_with_noise.pgf'):
    """
    Create and plot the quantum circuit with noise channels.

    Args:
        num_qubits: Number of qubits (default 4 for ski)
        num_layers: Number of variational layers (default 2 for ski)
        depolarizing_p: Depolarizing noise parameter
        save_path: Path to save the figure
    """
    # Create quantum device for mixed states (supports noise)
    dev = qml.device('default.mixed', wires=num_qubits)

    params_per_layer = num_qubits * 3
    total_params = params_per_layer * num_layers

    @qml.qnode(dev)
    def quantum_circuit_with_noise(state_amplitudes, params):
        """Quantum circuit with noise channels."""
        # 1. Amplitude encoding
        qml.StatePrep(state_amplitudes, wires=range(num_qubits))

        # 2. Variational layers with noise
        param_idx = 0
        for layer in range(num_layers):
            # Rotation layer with noise after each gate
            for qubit in range(num_qubits):
                qml.RX(params[param_idx], wires=qubit)
                qml.DepolarizingChannel(depolarizing_p, wires=qubit)

                qml.RY(params[param_idx + 1], wires=qubit)
                qml.DepolarizingChannel(depolarizing_p, wires=qubit)

                qml.RZ(params[param_idx + 2], wires=qubit)
                qml.DepolarizingChannel(depolarizing_p, wires=qubit)

                param_idx += 3

            # Entangling layer with noise
            if layer < num_layers - 1:
                for qubit in range(num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                    qml.DepolarizingChannel(depolarizing_p, wires=qubit)
                    qml.DepolarizingChannel(depolarizing_p, wires=qubit + 1)

                # Ring topology
                qml.CNOT(wires=[num_qubits - 1, 0])
                qml.DepolarizingChannel(depolarizing_p, wires=num_qubits - 1)
                qml.DepolarizingChannel(depolarizing_p, wires=0)

        # 3. Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    print(f"\n{'='*70}")
    print(f"Quantum REINFORCE Circuit with Noise (Ski Environment)")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Number of qubits: {num_qubits}")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Depolarizing noise: p = {depolarizing_p}")

    # Create example state and parameters
    state_size = 2 ** num_qubits
    example_state = np.zeros(state_size)
    example_state[0] = 1.0
    example_params = np.random.randn(total_params) * 0.01

    # Draw the circuit (decimals=None hides parameter values)
    fig, ax = qml.draw_mpl(quantum_circuit_with_noise, decimals=None, style='pennylane')(
        example_state, example_params
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure in both PGF and PNG formats
    pgf_path = save_path.replace('.png', '.pgf')
    png_path = save_path.replace('.pgf', '.png')

    # Save PNG first
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

    # Try to save PGF (may fail with Unicode characters from PennyLane)
    try:
        plt.savefig(pgf_path, format='pgf', bbox_inches='tight')
        print(f"\n✓ Noisy circuit diagram saved to:")
        print(f"  - {pgf_path} (PGF for LaTeX)")
        print(f"  - {png_path} (PNG preview)")
    except Exception as e:
        # PGF export failed - save as PDF instead which can be included in LaTeX
        pdf_path = pgf_path.replace('.pgf', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"\n✓ Noisy circuit diagram saved to:")
        print(f"  - {pdf_path} (PDF for LaTeX - use \\includegraphics)")
        print(f"  - {png_path} (PNG preview)")
        print(f"  Note: PGF export failed due to Unicode characters. PDF is a better option for LaTeX.")

    plt.close()

    print(f"\nNote: Depolarizing channels are applied after each gate operation.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Plot clean circuit (default ski configuration)
    print("\nGenerating quantum circuit visualization for ski environment...")
    plot_circuit(num_qubits=4, num_layers=2, save_path='quantum_circuit_ski.pgf')

    # Plot circuit with noise (as used in model checking)
    print("\nGenerating noisy quantum circuit visualization...")
    plot_circuit_with_noise(
        num_qubits=4,
        num_layers=2,
        depolarizing_p=0.01,
        save_path='quantum_circuit_ski_with_noise.pgf'
    )

    print("\n✓ All circuit diagrams generated successfully!")
    print("\nNote: PGF export is not supported due to Unicode characters in PennyLane's circuit labels.")
    print("PDF files have been generated instead, which work perfectly with LaTeX using \\includegraphics.")
