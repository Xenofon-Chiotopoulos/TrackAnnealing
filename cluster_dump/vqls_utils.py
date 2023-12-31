import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
#pauli_decomp = qml.pauli_decompose(A_num,pauli=True) # uncomment to see matrix decomposition in pauli matrices but this is used in the definition put in pennylane_vqls.py scipt

def make_CA(idx: int, ancilla_idx: int, pauli_strings: list[list[str]]) -> None:
    """Constructs the controlled unitary component A_l of the problem matrix A.

    Args:
        idx (int): Index of the Pauli string in the Pauli string list.
        ancilla_idx (int): Index of the ancilla qubit.
        pauli_string (list): List of Pauli strings.

    """
    # Iterate over the Pauli string and apply the appropriate controlled operations
    for id in range(len(pauli_strings)):
        if id == idx:
            sub_pauli = pauli_strings[id]
            for gate_index ,gate in enumerate(sub_pauli):
                # Apply Identity, CNOT or CZ based on the Pauli symbol
                if gate == "I":
                    continue  
                elif gate == "X":
                    qml.CNOT(wires=[ancilla_idx, gate_index])
                elif gate == "Y":
                    qml.CY(wires=[ancilla_idx, gate_index])
                elif gate == "Z":
                    qml.CZ(wires=[ancilla_idx, gate_index])

def construct_matrix(c: list[float], pauli_strings: list[list[str]]) -> np.tensor:
    """
    Construct a matrix given coefficients and Pauli strings.

    Parameters:
    - c (array): Coefficients for each Pauli term.
    - pauli_strings (list): List of tuples representing Pauli strings.

    Returns:
    - np.array: Constructed matrix.
    """

    # Define Pauli matrices
    Id = np.identity(2)#,dtype='complex128')
    Z = np.array([[1, 0], [0, -1]])#,dtype='complex128')
    X = np.array([[0, 1], [1, 0]])#,dtype='complex128')
    Y = np.array([[0,complex(0,-1)], [complex(0,1), 0]])

    # Construct the matrix using vectorized operations
    matrix = 0
    for pauli_string, coeff in zip(pauli_strings, c):
        pauli_term = 0
        for gate_index, gate in enumerate(pauli_string):
            if gate_index == 0:
                if gate == "I":
                    pauli_term = Id
                elif gate == "X":
                    pauli_term = X
                elif gate == "Y":
                    pauli_term = Y
                elif gate == "Z":
                    pauli_term = Z
            else:
                if gate == "I":
                    pauli_term = np.kron(pauli_term, Id)
                elif gate == "X":
                    pauli_term = np.kron(pauli_term, X)
                elif gate == "Y":
                    pauli_term = np.kron(pauli_term, Y)
                elif gate == "Z":
                    pauli_term = np.kron(pauli_term, Z)
        matrix += coeff * pauli_term
    return matrix

def calc_c_probs(A_num: np.tensor, b: np.tensor) -> np.tensor:
    """
    Calculates the vector `x` that satisfies the equation `Ax = b`.

    Args:
        A (pennylane.numpy.tensor.tensor): tensor representation of the matrix `A`.
        b (numpy.ndarray): NumPy array representation of the vector `b`.

    Returns:
        pennylane.numpy.tensor.tensor: Calculated vector `x`.
    """
    A_inv = np.linalg.inv(A_num)
    x = np.dot(A_inv, b)
    c_probs = (x / np.linalg.norm(x)) ** 2
    return c_probs

def process_vqls_output(n_qubits: int, w: np.tensor, ansatz: list[dict], A: np.tensor, b: np.array, n_shots: int, print_output = False) -> np.tensor:
    """
    Process the output of a variational quantum linear system algorithm (VQLS).

    Args:
        n_qubits (int): Number of qubits.
        w (pennylane.numpy.tensor.tensor): Coefficients of the Pauli operators.
        ansatz (list[dict]): Ansatz circuit description.
        A (pennylane.numpy.tensor.tensor): Pauli operators matrix.
        b (numpy.array): Vector of coefficients for each Pauli term.
        n_shots (int): Number of shots for the quantum simulation.
        print: (bool, optional): Whether to print the calculated probabilities. Defaults to False.

    Returns:
        tuple[pennylane.numpy.tensor.tensor, pennylane.numpy.tensor.tensor]: Calculated classical and quantum probabilities.
    """
    dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)

    @qml.qnode(dev_x, interface="autograd")
    def prepare_and_sample(weights):
        custom_variational_block(weights, ansatz)
        return qml.sample()

    c_probs = calc_c_probs(A, b)

    raw_samples = prepare_and_sample(w)

    samples = []
    for sam in raw_samples:
        samples.append(int("".join(str(bs) for bs in sam), base=2))

    q_probs = np.bincount(samples) / n_shots

    c_probs = 1/np.max(c_probs)*c_probs
    q_probs = 1/np.max(q_probs)*q_probs

    if print_output == True:
        print("x_n^2 =\n", c_probs)
        print("|<x|n>|^2=\n", q_probs)
        print('Difference=\n', c_probs-q_probs)

    return c_probs, q_probs

def variational_block(weights: np.tensor, n_qubits: int, params=None) -> None:
        """
        Old custom set variational block
        """
        for idx in range(n_qubits):
            qml.Hadamard(wires=idx)

        for idx, element in enumerate(weights):
            if idx < n_qubits:
                if idx % 2 == 0:
                    qml.RY(element, wires=idx)
            if idx >= n_qubits:
                idx = idx - n_qubits
                qml.RZ(element, wires=idx)
        
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

def count_variational_gates(ansatz: list[dict], all_gates = False) -> int:
    """
    Counts the number of variational gates in the provided ansatz circuit.

    Args:
        ansatz (list[dict]): List of gate information dictionaries.

    Returns:
        int: Total number of variational gates in the ansatz.
    """
    variational_gate_count = 0
    non_variational_gate_count = 0
    for gate_info in ansatz:
        if gate_info["gate"] == "RX" or gate_info["gate"] == "RY" or gate_info["gate"] == "RZ" or gate_info["gate"] == "P":
            variational_gate_count += len(gate_info["wires"])

        if gate_info["gate"] == "CNOT" or gate_info["gate"] == "H" :
            non_variational_gate_count += len(gate_info["wires"])

    if all_gates == True:
        return variational_gate_count, non_variational_gate_count
    else:
        return variational_gate_count

def custom_variational_block(weights: np.tensor, ansatz: list[dict]) -> None:
    """
    Implements a custom variational block using the provided ansatz.

    Args:
        weights: pennylane.numpy.tensor.tensor representing the weights of the variational parameters.
        ansatz: List of gate information dictionaries describing the variational circuit.

    Returns:
        None

    """
    weight_index = 0
    for gate_info in ansatz:
        gate_name = gate_info["gate"]
        wire_indices = gate_info["wires"]

        for wire in wire_indices:
            if gate_name == "H":
                qml.Hadamard(wires=wire)
            elif gate_name == "RX":
                qml.RX(weights[weight_index],wires=wire)
                weight_index += 1
            elif gate_name == "RY":
                qml.RY(weights[weight_index],wires=wire)
                weight_index += 1
            elif gate_name == "RZ":
                qml.RZ(weights[weight_index],wires=wire)
                weight_index += 1
            elif gate_name == "P":
                qml.PhaseShift(weights[weight_index],wires=wire)
                weight_index += 1
            elif gate_name == "CNOT":
                qml.CNOT(wires=wire)
                

def run_vqls(n_qubits: int, pauli_string: list[list[str]], c: list[float], ansatz: list[dict], rng_seed = 0, steps = 30, eta = 0.8 ,q_delta = 0.1 , print_step = True) -> np.tensor:
    tot_qubits = n_qubits + 1  
    ancilla_idx = n_qubits 

    def U_b():
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        for idx in range(n_qubits):
            qml.Hadamard(wires=idx)

    def CA(idx,params=None):
        make_CA(idx,ancilla_idx,pauli_string)

    dev_mu = qml.device("lightning.qubit", wires=tot_qubits)

    @qml.qnode(dev_mu, interface="autograd")
    def local_hadamard_test(weights, l=None, lp=None, j=None, part=None , params=None):

        qml.Hadamard(wires=ancilla_idx)

        if part == "Im" or part == "im":
            qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

        custom_variational_block(weights, ansatz)

        CA(l,params)

        U_b()

        if j != -1:
            qml.CZ(wires=[ancilla_idx, j])

        U_b()

        CA(lp,params)

        qml.Hadamard(wires=ancilla_idx)

        # Expectation value of Z for the ancillary qubit.
        return qml.expval(qml.PauliZ(wires=ancilla_idx))

    def mu(weights, l=None, lp=None, j=None, params=None):
        """Generates the coefficients to compute the "local" cost function C_L."""

        mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re", params=params)
        mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im", params=params)

        return mu_real + 1.0j * mu_imag

    def psi_norm(weights, params=None):
        """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
        norm = 0.0

        for l in range(0, len(c)):
            for lp in range(0, len(c)):
                norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1,params)

        return abs(norm)

    def cost_loc(weights,params=None):
        """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
        mu_sum = 0.0

        for l in range(0, len(c)):
            for lp in range(0, len(c)):
                for j in range(0, n_qubits):
                    mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j,params)

        mu_sum = abs(mu_sum)

        # Cost function C_L
        return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))

    np.random.seed(rng_seed)
    w = q_delta * np.random.randn(count_variational_gates(ansatz), requires_grad=True)

    opt = qml.GradientDescentOptimizer(eta)

    cost_history = []
    for it in range(steps):
        w, cost = opt.step_and_cost(cost_loc, w)
        if print_step == True : print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
        if len(cost_history) > 0:
            if np.round(cost_history[-1],6) == np.round(cost,6) and cost < 0.000001:
                break
        cost_history.append(cost)
    return w, cost_history

def plot_vqls_results(n_qubits: int, c_probs: list[float], q_probs: list[float], cost_history: list[np.tensor]) -> None:
    """
    Plots the results of a variational quantum linear system algorithm (VQLS) using classical and quantum probabilities.

    Args:
        n_qubits (int): Number of qubits in the system.
        c_probs (list[float]): List of classical probabilities.
        q_probs (list[float]): List of quantum probabilities.
        cost_history (list[pennylane.numpy.tensor.tensor]): List of cost function values during optimization.

    Returns:
        None
    """
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
    plt.savefig("cluster_dump/quantum_probabilities.png")