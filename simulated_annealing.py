import matplotlib.pyplot as plt
from dp_hamiltonian import generate_hamiltonian
import numpy as np
import dimod
import time

def simulated_annealing(event, params, plot=False):

    if plot == True:
        fig = plt.figure()
        fig.set_size_inches(12, 6)
        ax = plt.axes(projection='3d')
        event.display(ax)
        ax.view_init(vertical_axis='y')
        fig.set_tight_layout(True)
        ax.axis('off')
        ax.set_title(f"Generated event\n{len(event.modules)} modules\n{len(event.tracks)} tracks - {len(event.hits)} hits")
        plt.show()

    start = time.time()
    A, b, components, segments = generate_hamiltonian(event, params)

    end = time.time()
    ham_time = end - start
    print(end - start,'s for Hamiltonian Generation')

    if plot == True:
        fig = plt.figure()
        fig.set_size_inches(12, 6)
        ax = plt.axes(projection='3d')
        event.display(ax, show_tracks=False)

        for segment in segments:
            segment.display(ax)

        ax.view_init(vertical_axis='y')
        fig.set_tight_layout(True)
        ax.axis('off')

        ax.set_title(f"{len(segments)} segments generated")

        plt.show()

    # Define the BQM and sampler for simulated annealing
    start = time.time()

    offset = 0.0
    vartype = dimod.BINARY
    bqm= dimod.BinaryQuadraticModel(b, A, offset, vartype)
    sampler = dimod.SimulatedAnnealingSampler()
    #can use everything above redo bqm
    #-----------------------------------------------------------------------------------------------

    # Run simulated annealing and retrieve the best sample
    response = sampler.sample(bqm, num_reads=1000)
    best_sample = response.first.sample
    print(best_sample)
    sol_sample = np.array(list(best_sample.values()))
    print(response.first.energy)

    end = time.time()
    print(end - start,'s for simulated annealing')
    annealing_time = end - start

    solution_segments = [seg for sol, seg in zip(sol_sample, segments) if sol == 1]

    # Check if there are any segments in the solution
    if len(solution_segments) == 0:
        print("No segments included in the solution.")
    else:
    # Display the solution
        if plot == True:
            fig = plt.figure()
            fig.set_size_inches(12, 6)
            ax = plt.axes(projection='3d')
            event.display(ax, show_tracks=False)

            for segment in solution_segments:
                segment.display(ax)

            ax.view_init(vertical_axis='y')
            fig.set_tight_layout(True)
            ax.axis('off')
            ax.set_title(f"Solution")
            plt.show()
    if plot == True:
        fig, axs = plt.subplots(2,3)
        fig.set_size_inches(10,6)
        vmin = np.min([A.min()].extend(components[key].min() for key in components))
        vmax = np.max([A.max()].extend(components[key].max() for key in components))
        im = axs[0,0].matshow(A,vmin=vmin, vmax=vmax)
        axs[0,0].set_title("A")

        axs_raviter = iter(axs.ravel())
        next(axs_raviter)
        for key in components:
            ax = next(axs_raviter)
            ax.matshow(components[key],vmin=vmin, vmax=vmax)
            ax.set_title(key)

        fig.colorbar(im, ax=axs.ravel().tolist())
        plt.show()
    
    reutrn_dict = {
        'Hamiltonian_time':ham_time,
        'Annealing_time':annealing_time
    }
    return ham_time, annealing_time