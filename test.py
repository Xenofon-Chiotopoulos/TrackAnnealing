import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info import operators, Pauli
from qiskit.quantum_info.operators import Operator

A_3 = [[0, 1, 0, 0],
       [1, 0, 0, 0],
       [0, 0, 0,-1],
       [0, 0,-1, 0]]

A_3 = [[1, 0, 0, 0],
       [0,-1, 0, 0],
       [0, 0,-1, 0],
       [0, 0, 0, 1]]

XZ = Operator(A_3)

controls = QuantumRegister(2)
qc = QuantumCircuit(controls)
qc.unitary(XZ, [0, 1], label='XZ')
print(qc)
qc = qc.decompose()
print(qc)
# Initialize a quantu
# m circuit with two qubits
'''
qc = QuantumCircuit(4)

qc.cx(3, 0)

# Apply the CZ gate to the control qubit (ancilla_idx) and the target qubit (1)
qc.cz(3, 1)


print(qc.draw(output='text'))
qc = qc.decompose()

# Print the circuit
print(qc.draw(output='text'))
'''

import pennylane as qml
import numpy as np

A_3 = np.array([[0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, -1],
                [0, 0, -1, 0]])

A_4 = np.array([[1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]])

# Define the quantum circuit
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit_A_3():
    qml.QubitUnitary(A_3, wires=[0, 1])
    return qml.state()

@qml.qnode(dev)
def circuit_A_4():
    qml.QubitUnitary(A_4, wires=[0, 1])
    return qml.state()

result = circuit_A_3()
print("State for A_3:", result)
print("Circuit for A_3:")
print(qml.draw(circuit_A_3)())

result = circuit_A_4()
print("State for A_4:", result)
print("Circuit for A_4:")
print(qml.draw(circuit_A_4)())



