"""
This script is an example illustrating the process of estimating the accuracy of the pUCCD algorithm on mock hydrogen chains. 
The script simulates hydrogen chains of different lengths (num_qubits), constructs the corresponding pUCCD circuits, and then computes their expectation values using a noiseless simulation.
"""
from pathlib import Path

import numpy as np
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow.primitive_ops import PauliSumOp

import ansatz
import simulator
from ansatz import PUCCD
from simulator import Simulator
import common
import sys

sys.path[1:1] = ["Dev", "hydrogen-lattice/Dev"]
sys.path[1:1] = ["../../Dev", "../../Dev", "../../hydrogen-lattice/Dev/"]
sys.path[1:1] = ["_common", "_common/qiskit", "hydrogen-lattice/_common"]
sys.path[1:1] = [
    "../../_common",
    "../../_common/qiskit",
    "../../hydrogen-lattice/_common/",
]


# Create an instance of the Simulator class for noiseless simulations
ideal_backend = simulator.Simulator()

# Initialize an empty list to accumulate simulation data for hydrogen chains of different lengths
simulation_data = []

# Instantiate the pUCCD algorithm
puccd = PUCCD()

# Define the number of shots (number of repetitions of each quantum circuit)
# For the noiseless simulation, we use 10,000 shots.
# For the statevector simulator, this would be set to None.
shots = 10_000

def compute_energy(circuit, operator, shots):
    # Compute the expectation value of the circuit with respect to the Hamiltonian for optimization
    ideal_energy = ideal_backend.compute_expectation(
        circuit, operator=operator, shots=shots
    )
    return ideal_energy

#get a list of all the problem and solution paths
directory_path = Path("../_common/instances")
problems = list(directory_path.glob("**/*.json"))
solutions = list(directory_path.glob("**/*.sol"))

# Print the list of JSON file paths
print(f"problem files: {problems}")
print(f"solution files: {solutions}")

# get the number of qubits- should be roughly in the same spot in every file
num_qubits = [int(problem.name.split("_")[0][1:]) for problem in problems]
print(f"num_qubits: {num_qubits}")

#get the PauliSumOps from problems
pauli_ops = []

for file_path in problems:
    ops, coefs = common.read_paired_instance(file_path)

    hamiltonians = list(zip(ops, coefs))

    pauli_ops.append(PauliSumOp.from_list(hamiltonians))

# Loop over hydrogen chains with different numbers of qubits (from 2 to 4 in this example)
# These hydrogen chains are specified via the problem files
for index, pauli_op in enumerate(pauli_ops):
    print(f"Starting optimization for problem specified in {problems[index]}...")

    # Construct the pUCCD circuit for the current mock hydrogen chain
    circuit = puccd.build_circuit(num_qubits[index])

    operator = pauli_op

    # Initialize the parameters with -1e-3 or 1e-3
    initial_parameters = [
        np.random.choice([-1e-3, 1e-3]) for _ in range(len(circuit.parameters))
    ]
    circuit.assign_parameters(initial_parameters, inplace=True)

    # Initialize the COBYLA optimizer
    optimizer = COBYLA(maxiter=1000, tol=1e-6, disp=False)

    # Optimize the circuit parameters using the optimizer
    optimized_parameters = optimizer.minimize(
        lambda parameters: compute_energy(circuit, operator, shots=shots),
        x0=np.random.choice([-1e-3, 1e-3]),
    )

    # Extract the parameter values from the optimizer result
    optimized_values = optimized_parameters.x

    # Create a dictionary of {parameter: value} pairs
    parameter_values = {
        param: value for param, value in zip(circuit.parameters, optimized_values)
    }

    # Assign the optimized values to the circuit parameters
    circuit.assign_parameters(parameter_values, inplace=True)

    ideal_energy = ideal_backend.compute_expectation(
        circuit, operator=operator, shots=shots
    )

    print(f"PUCCD calculated energy: {ideal_energy}")

    solution = solutions[index]

    method_names, values = common.read_puccd_solution(solution)

    for method, value in zip(method_names, values):
        print(f"{method}: {value}")
