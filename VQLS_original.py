import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import ArbitraryUnitary
import matplotlib.pyplot as plt

n_qubits = 3  # Number of system qubits.
n_shots = 10 ** 6  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 30  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.1  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator

c = np.array([3.0, -0.5, -0.5])
Id = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

A_0 = np.identity(8)
A_1 = np.kron(np.kron(X, Id), Id)
A_2 = np.kron(np.kron(X, Z), Z)

A_num = c[0] * A_0 + c[1] * A_1 + c[2] * A_2
b = np.ones(8) / np.sqrt(8)
pauli_decomp = qml.pauli_decompose(A_num,pauli=True)
print(A_num)
print(pauli_decomp)


def U_b():
    """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

def CA(idx):
    """Controlled versions of the unitary components A_l of the problem matrix A."""
    if idx == 0:
        None

    elif idx == 1:
        qml.CNOT(wires=[ancilla_idx, 0])
        
    elif idx == 2:
        qml.CNOT(wires=[ancilla_idx, 0])
        qml.CZ(wires=[ancilla_idx, 1])
        qml.CZ(wires=[ancilla_idx, 2])


def variational_block(weights):
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    for idx, element in enumerate(weights):
        if idx < n_qubits:
            qml.RY(element, wires=idx)
        if idx >= n_qubits:
            idx = idx - 3
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

    # Expectation value of Z for the ancillary qubit.
    return qml.expval(qml.PauliZ(wires=ancilla_idx))

def mu(weights, l=None, lp=None, j=None):
    """Generates the coefficients to compute the "local" cost function C_L."""

    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag

def psi_norm(weights):
    """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)

    return abs(norm)

def cost_loc(weights):
    """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)

    mu_sum = abs(mu_sum)

    # Cost function C_L
    return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))

np.random.seed(rng_seed)
w = q_delta * np.random.randn(n_qubits, requires_grad=True)

opt = qml.GradientDescentOptimizer(eta)

cost_history = []
for it in range(steps):
    w, cost = opt.step_and_cost(cost_loc, w)
    print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
    cost_history.append(cost)

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



plt.style.use("ggplot")  
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Classical probabilities
ax1.bar(np.arange(0, 2 ** n_qubits), c_probs, color="blue")
ax1.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax1.set_xlabel("Vector space basis")
ax1.set_title("Classical probabilities")

# Plot 2: Quantum probabilities
ax2.bar(np.arange(0, 2 ** n_qubits), q_probs, color="green")
ax2.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Quantum probabilities")

# Plot 3: Cost function optimization
ax3.plot(cost_history, color='green', marker='o', linestyle='-', linewidth=2, markersize=8, label='Cost Function')
ax3.set_title("Optimization Progress")
ax3.set_xlabel("Optimization Steps")
ax3.set_ylabel("Cost Function Value")

plt.tight_layout()
plt.show()


