import matplotlib.pyplot as plt
from dp_hamiltonian import generate_hamiltonian
import numpy as np
import dimod
import time

def simulated_annealing(event, params, num_reads, annealing_type='BQM', plot=False):

    start = time.time()
    A, b, components, segments = generate_hamiltonian(event, params)

    end = time.time()
    ham_time = end - start
    print(end - start,'s for Hamiltonian Generation')

    # Define the BQM and sampler for simulated annealing
    start = time.time()

    offset = 0.0
    vartype = dimod.BINARY
    bqm= dimod.BinaryQuadraticModel(b, A, offset, vartype)

    Q, off = bqm.to_qubo()

    sampler = dimod.SimulatedAnnealingSampler()

    #can use everything above redo bqm
    #-----------------------------------------------------------------------------------------------

    # Run simulated annealing and retrieve the best sample
    if annealing_type == 'QUBO':
        response = sampler.sample_qubo(Q, num_reads=num_reads)
    if annealing_type == 'BQM':
        response = sampler.sample(bqm, num_reads=num_reads)
    if annealing_type == 'ISING':
        response = sampler.sample_ising(A, b, num_reads=num_reads)

    best_sample = response.first.sample
    #print(best_sample)
    sol_sample = np.array(list(best_sample.values()))
    #print(response.first.energy)

    end = time.time()
    print(end - start,'s for simulated annealing')
    annealing_time = end - start

    solution_segments = [seg for sol, seg in zip(sol_sample, segments) if sol == 1]

    # Check if there are any segments in the solution
    if len(solution_segments) == 0:
        print("No segments included in the solution.")
    
    reutrn_dict = {
        'Hamiltonian_time':ham_time,
        'Annealing_time':annealing_time
    }

    return [ham_time, annealing_time, event, segments, solution_segments, A, components]