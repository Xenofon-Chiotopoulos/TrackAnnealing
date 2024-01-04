# Pennylane
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import trackhhl.hamiltonians.simple_hamiltonian as hamiltonian
import trackhhl.toy.simple_generator as toy

N_MODULES = 4
N_PARTICLES = 3
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
print(ham.A.todense())
print(ham.b)
A_inv = np.linalg.inv(ham.A.todense())
x = np.dot(A_inv, ham.b)
c_probs = (x / np.linalg.norm(x)) ** 2


n_qubits = 3  # Number of system qubits.
n_shots = 10 ** 6  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 30  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.001  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator

c = np.array([3.0, -0.5, -0.5])

mat_size = 8
Id = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
A_0 = np.identity(mat_size)
A_1 = np.kron(np.kron(Z, Z), X)
A_2 = np.kron(np.kron(X, Id), Id)

A_num = c[0] * A_0 + c[1] * A_1 + c[2] * A_2 


A = ham.A.todense()
A = A_num
pauli_decomp = qml.pauli_decompose(A,pauli=True)

print('\n','\n')
print(pauli_decomp)


def U_b():
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

def CA(idx):
    if idx == 0:
        None

    elif idx == 1:
        qml.CNOT(wires=[ancilla_idx, 0])
    
    elif idx == 2:
        qml.CZ(wires=[ancilla_idx, 0])
        qml.CZ(wires=[ancilla_idx, 1])
        qml.CNOT(wires=[ancilla_idx, 2])


def variational_block(weights):
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    for idx, element in enumerate(weights):
        if idx < n_qubits:
            qml.RY(element, wires=idx)
        if idx >= n_qubits:
            idx = idx - n_qubits
            qml.RY(element, wires=idx)
    
        
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
       
    
dev_mu = qml.device("lightning.qubit", wires=tot_qubits)

@qml.qnode(dev_mu, interface="autograd")
def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

    qml.Hadamard(wires=ancilla_idx)
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)
    variational_block(weights)

    CA(l)

    U_b()

    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])

    U_b()

    CA(lp)

    qml.Hadamard(wires=ancilla_idx)

    return qml.expval(qml.PauliZ(wires=ancilla_idx))

def mu(weights, l=None, lp=None, j=None):

    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag


def psi_norm(weights):
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)

    return abs(norm)

def cost_loc(weights):
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)

    mu_sum = abs(mu_sum)

    return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))



np.random.seed(rng_seed)
w = q_delta * np.random.randn(n_qubits, requires_grad=True)

opt = qml.GradientDescentOptimizer(eta)

cost_history = []
for it in range(steps):
    w, cost = opt.step_and_cost(cost_loc, w)
    print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
    cost_history.append(cost)


plt.style.use("classic")
plt.plot(cost_history, "g")
plt.ylabel("Cost function")
plt.xlabel("Optimization steps")
plt.show()
'''
c1 = np.array([1.0,0.2,0.2])
mat_size = 8
Id = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

A_0 = np.identity(mat_size)
A_1 = np.kron(np.kron(X, Z), Id)
A_2 = np.kron(np.kron(X, Id), Id)

A_num = c1[0] * A_0 + c1[1] * A_1 + c1[2] * A_2 

b = np.ones(mat_size) / np.sqrt(mat_size)

pauli_decomp = qml.pauli_decompose(A_num,pauli=True)
#print(A_num)
#print(pauli_decomp)
'''
A_num = A
b = ham.b



print("A = \n", A_num)
print("b = \n", b)

A_inv = np.linalg.inv(A_num)
x = np.dot(A_inv, b)
c_probs = (x / np.linalg.norm(x)) ** 2

dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)

@qml.qnode(dev_x, interface="autograd")
def prepare_and_sample(weights):
    variational_block(weights)
    return qml.sample()

raw_samples = prepare_and_sample(w)

samples = []
for sam in raw_samples:
    samples.append(int("".join(str(bs) for bs in sam), base=2))

q_probs = np.bincount(samples) / n_shots

print("x_n^2 =\n", c_probs)
print("|<x|n>|^2=\n", q_probs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))

ax1.bar(np.arange(0, 2 ** n_qubits), c_probs, color="blue")
ax1.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax1.set_xlabel("Vector space basis")
ax1.set_title("Classical probabilities")

ax2.bar(np.arange(0, 2 ** n_qubits), q_probs, color="green")
ax2.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Quantum probabilities")

plt.show()