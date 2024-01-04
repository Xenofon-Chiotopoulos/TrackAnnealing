import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import ArbitraryUnitary
import matplotlib.pyplot as plt
from vqls_utils import construct_matrix, plot_vqls_results, run_vqls, process_vqls_output,count_variational_gates
import time

n_qubits = 3  # Number of system qubits.
n_shots = 10 ** 6  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 31  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.1  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator

pauli_string=[["X","Z","I"],["X","Z","X"],["X","I","Z"]]
ansatz = [
    #{"gate": "H", "wires": [0,1,2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    #{"gate": "CNOT", "wires": [[0,1], [1,2]]},
]

c = np.array([1.0, 1.0, 1.0]) # the cost func convergence is not relevant to the coeff array so we will set to 1 for all
params = []
A_num = construct_matrix(c,pauli_string)
b = np.ones(8) 

start = time.time()
w, cost_history = run_vqls(n_qubits, pauli_string, c, ansatz, rng_seed = 0, steps = 30, eta = 1.0 ,q_delta = 0.1, print_step = True)
end = time.time()
variational_gate_count, non_variational_gate_count = count_variational_gates(ansatz, all_gates=True)
print('VQLS time taken: ', end - start, 's\n', f'{variational_gate_count} variational gates')
print(f'{non_variational_gate_count} non-variational gates \n',f'{non_variational_gate_count+ variational_gate_count} total gates')

#c_probs, q_probs = process_vqls_output(n_qubits, w, ansatz, A_num, b, n_shots, print_output = False)

#plot_vqls_results(n_qubits, c_probs, q_probs, cost_history)

"""
timing tests: 
30 convergence steps:

variational:[1,2,3,6,9,12,24,36,48]
time: [2.928,5.26,6.09,12.063,19.651,28.444,81.571, 176.799, 269.197]

non-variational: [1,2,12,24,36, 48]
time: [0.897,0.923,1.179,1.352, 1.77, 1.84]
"""