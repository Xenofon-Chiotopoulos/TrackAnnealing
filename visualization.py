import numpy as np
import matplotlib.pyplot as plt
import trackhhl.hamiltonians.simple_hamiltonian as hamiltonian
import trackhhl.toy.simple_generator as toy

def make_matrix(N_MODULES=3,N_PARTICLES=2):

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
    A = ham.A.todense()
    return A

def visualize_matrix(matrix):
    fig, ax = plt.subplots()

    # Create a colormap for visualization
    cmap = plt.get_cmap('viridis')

    # Normalize the matrix values for color mapping
    norm = plt.Normalize(matrix.min(), matrix.max())

    # Create a colored matrix plot
    cax = ax.matshow(matrix, cmap=cmap, norm=norm)

    # Add annotations to each cell
    #for i in range(matrix.shape[0]):
    #    for j in range(matrix.shape[1]):
    #        if matrix[i, j] == -5:
    #            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white' if matrix[i, j] < matrix.max()/2 else 'black')

    # Add colorbar for reference
    cbar = fig.colorbar(cax)
    ax.xaxis.set_ticks_position('bottom')
    # Show the plot
    plt.title('The Matrix')
    plt.show()

A = make_matrix(3,2)
visualize_matrix(A)