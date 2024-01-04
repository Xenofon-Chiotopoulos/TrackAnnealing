import trackhhl.hamiltonians.simple_hamiltonian as hamiltonian
import trackhhl.toy.simple_generator as toy
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import pennylane as qml
from my_qulacs.utils import *
import time

start = time.time()

N_MODULES = 3
N_PARTICLES = 4
LX = float("+inf")
LY = float("+inf")
Z_SPACING = 1.0

detector = toy.SimpleDetectorGeometry(
    module_id=list(range(N_MODULES)),
    lx=[LX]*N_MODULES,
    ly=[LY]*N_MODULES,
    z=[i+Z_SPACING for i in range(N_MODULES)])

generator = toy.SimpleGenerator(
    detector_geometry=detector,
    theta_max=np.pi/6)
event = generator.generate_event(N_PARTICLES)
ham = hamiltonian.SimpleHamiltonian(
    epsilon=1e-3,
    gamma=2.0,
    delta=1.0)
ham.construct_hamiltonian(event=event)
#print(ham.A.todense())
print(np.shape(ham.A.todense()))
print(np.shape(ham.b))

A = ham.A.todense()
b = ham.b
#pauli_decomp = qml.pauli_decompose(A,pauli=True)
#print(pauli_decomp)

x_exact = np.linalg.lstsq(A, b, rcond=0)[0]


def pad_to_power_of_two(matrix):
    # Get the current shape of the matrix
    num_rows, num_cols = matrix.shape

    # Calculate the next power of 2 for both rows and columns
    padded_rows = 2**int(np.ceil(np.log2(num_rows)))
    padded_cols = 2**int(np.ceil(np.log2(num_cols)))

    # Pad the matrix with zeros to the calculated size
    padded_matrix = np.zeros((padded_rows, padded_cols), dtype=matrix.dtype)
    padded_matrix[:num_rows, :num_cols] = matrix

    return padded_matrix

#A_test = pad_to_power_of_two(A)
#print(np.shape(A_test))
#A = A_test

####################################################################################################################################################################################################
# number of registers used for phase estimation
reg_nbit = 3
nbit = 5  ## Number of bits used for state
N = 2**nbit

## Factor to scale W_enl
scale_fac = 1.
W_enl_scaled = scale_fac * A

## Minimum value to be assumed as an eigenvalue of W_enl_scaled
## In this case, since the projection succeeds 100%, we set the value as a constant multiple of the minimum value that can be represented by the register.
C = 0.5*(2 * np.pi * (1. / 2**(reg_nbit) ))

## diagonalization. AP = PD <-> A = P*D*P^dag
D, P = np.linalg.eigh(W_enl_scaled)

#####################################
### Create an HHL quantum circuit. Starting from the 0th bit,
### we have the bits in the space where A acts (0th ~ nbit-1th),
### the register bits (nbit th ~ nbit+reg_nbit-1 th), and
### the bits for conditional rotation (nbit+reg_nbit th).
#####################################

total_qubits = nbit + reg_nbit + 1
total_circuit = QuantumCircuit(total_qubits)

## ------ Prepare vector b input to the 0th~(nbit-1)th bit ------
## Normally we should use qRAM algorithm, but here we use our own input gate.
## In qulacs, you can also implement it with state.load(b_enl).
state = QuantumState(total_qubits)
state.set_zero_state()
b_gate = input_state_gate(0, nbit-1, b)
total_circuit.add_gate(b_gate)

## ------- Hadamard gate on register bit -------
for register in range(nbit, nbit+reg_nbit): ## from nbit to nbit+reg_nbit-1
    total_circuit.add_H_gate(register)

## ------- Implement phase estimation -------
## U := e^{i*A*t) and its eigenvalues are diag( {e^{i*2pi*phi_k}}_{k=0, ... N-1)).
## Implement \sum_j |j><j| exp(i*A*t*j) to register bits
for register in range(nbit, nbit+reg_nbit):
    ## Implement U^{2^{register-nbit}}.
    ## Use diagonalized results.
    U_mat = reduce(np.dot,  [P, np.diag(np.exp( 1.j * D * (2**(register-nbit)) )), P.T.conj()]  )
    U_gate = gate.DenseMatrix(np.arange(nbit), U_mat)
    U_gate.add_control_qubit(register, 1) ## add control bit
    total_circuit.add_gate(U_gate)

## ------- Perform inverse QFT to register bits -------
total_circuit.add_gate(QFT_gate(nbit, nbit+reg_nbit-1, Inverse=True))

## ------- multiply conditional rotation -------
## The eigenvalue of A*t corresponding to the register |phi> is l = 2pi * 0. phi = 2pi * (phi / 2**reg_nbit).
## The definition of conditional rotation is (opposite of the text)
## |phi>|0> -> C/(lambda)|phi>|0> + sqrt(1 - C^2/(lambda)^2)|phi>|1>.
## Since this is a classical simulation, the gate is made explicitly.

condrot_mat = np.zeros( (2**(reg_nbit+1), (2**(reg_nbit+1))), dtype=complex)
for index in range(2**reg_nbit):
    lam = 2 * np.pi * (float(index) / 2**(reg_nbit) )
    index_0 = index ## integer which represents |index>|0>
    index_1 = index + 2**reg_nbit ## integer which represents |index>|1>
    if lam >= C:
        if lam >= np.pi: ## Since we have scaled the eigenvalues in [-pi, pi] beforehand, [pi, 2pi] corresponds to a negative eigenvalue
            lam = lam - 2*np.pi
        condrot_mat[index_0, index_0] = C / lam
        condrot_mat[index_1, index_0] =   np.sqrt( 1 - C**2/lam**2 )
        condrot_mat[index_0, index_1] = - np.sqrt( 1 - C**2/lam**2 )
        condrot_mat[index_1, index_1] = C / lam

    else:
        condrot_mat[index_0, index_0] = 1.
        condrot_mat[index_1, index_1] = 1.
## Convert to DenseGate and implement
condrot_gate = gate.DenseMatrix(np.arange(nbit, nbit+reg_nbit+1), condrot_mat)
print(condrot_gate)
total_circuit.add_gate(condrot_gate)
'''
# Apply controlled rotation
for index in range(2**reg_nbit):
    lam = 2 * np.pi * (float(index) / 2**(reg_nbit))
    if lam >= C:
        if lam >= np.pi:
            lam = lam - 2 * np.pi
        control_qubit = index
        target_qubit = reg_nbit
        total_circuit.add_gate(gate.CNOT(control_qubit, target_qubit))
        total_circuit.add_gate(gate.RY(reg_nbit,2 * lam))
        total_circuit.add_gate(gate.CNOT(control_qubit, target_qubit))

# Simulate the circuit
total_circuit.update_quantum_state(state)
'''
## ------- Perform QFT to register bits -------
total_circuit.add_gate(QFT_gate(nbit, nbit+reg_nbit-1, Inverse=False))

## ------- Implement the inverse of phase estimation (U^\dagger = e^{-iAt}) -------
for register in range(nbit, nbit+reg_nbit): ## from nbit to nbit+reg_nbit-1
    ## implement {U^{\dagger}}^{2^{register-nbit}}
    ## use diagonalized results.
    U_mat = reduce(np.dot,  [P, np.diag(np.exp( -1.j* D * (2**(register-nbit)) )), P.T.conj()]  )
    U_gate = gate.DenseMatrix(np.arange(nbit), U_mat)
    U_gate.add_control_qubit(register, 1) ## add a control bit
    total_circuit.add_gate(U_gate)

## ------- act Hadamard gate on register bit -------
for register in range(nbit, nbit+reg_nbit):
    total_circuit.add_H_gate(register)

## ------- Project auxiliary bits to 0. Implemented in qulacs as a non-unitary gate -------
total_circuit.add_P0_gate(nbit+reg_nbit)

#####################################
### Run the HHL quantum circuit and retrieve the result
#####################################
total_circuit.update_quantum_state(state)
print(total_circuit)


## The 0th to (nbit-1)th bit corresponds to the calculation result |x>.
result = state.get_vector()[:2**nbit].real
x_HHL = result/C * scale_fac

T = 0.35
#print(x_HHL)
x_HHL_ = (x_HHL > T).astype(int)
x_exact_ = (x_exact > T).astype(int)
#print(x_exact)
print(np.mean(x_exact-x_HHL))
print(np.mean(x_exact_-x_HHL_))
end = time.time()
print("Total time take:",end-start,"s")

plt.style.use("ggplot")  
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 4))

# Plot 1: Classical probabilities
ax1.bar(np.arange(0, 2 ** nbit), x_exact, color="blue")
ax1.set_xlim(-0.5, 2 ** nbit - 0.5)
ax1.set_xlabel("Vector space basis")
ax1.set_title("Classical probabilities")

# Plot 2: Quantum probabilities
ax2.bar(np.arange(0, 2 ** nbit), x_HHL, color="green")
ax2.set_xlim(-0.5, 2 ** nbit - 0.5)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Quantum probabilities")

# Plot 1: Classical probabilities
ax3.bar(np.arange(0, 2 ** nbit), x_exact_, color="blue")
ax3.set_xlim(-0.5, 2 ** nbit - 0.5)
ax3.set_xlabel("Vector space basis")
ax3.set_title("Classical probabilities")

# Plot 2: Quantum probabilities
ax4.bar(np.arange(0, 2 ** nbit), x_HHL_, color="green")
ax4.set_xlim(-0.5, 2 ** nbit - 0.5)
ax4.set_xlabel("Hilbert space basis")
ax4.set_title("Quantum probabilities")

plt.tight_layout()
plt.show()


'''
c = np.array([3.0, -0.5, -0.5])
Id = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

A_0 = np.identity(8)
A_1 = np.kron(np.kron(X, Id), Id)
A_2 = np.kron(np.kron(X, Z), Z)

A_num = c[0] * A_0 + c[1] * A_1 + c[2] * A_2
b = np.ones(8) / np.sqrt(8)
'''