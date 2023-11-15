import toymodel_3d_scat as toy
from dp_hamiltonian import generate_hamiltonian
from simulated_annealing import simulated_annealing
import numpy as np
import pickle

def save_list_to_file(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

params = {
    'alpha': 1.0,
    'beta': 1.0,
    'lambda': 100.0,
}

N_MODULES = 4
#N_TRACKS = 20

Ham_time = []
annealing_time = []
event_list = []
segments = []
solution_segments = []
A = []
components = []
for N_TRACKS in range(1,25):
    LX = 2
    LY = 2
    SPACING = 1
    print("Number of tracks:", N_TRACKS)
    detector = toy.generate_simple_detector(N_MODULES, LX, LY, SPACING)
    event = toy.generate_event(detector, N_TRACKS, theta_max=np.pi / 50, seed=1)

    sim_annealing = simulated_annealing(event, params, 1, 'QUBO')

    Ham_time.append(sim_annealing[0])
    annealing_time.append(sim_annealing[1])
    event_list.append(sim_annealing[2])
    segments.append(sim_annealing[3])
    solution_segments.append(sim_annealing[4])
    A.append(sim_annealing[5])
    components.append(sim_annealing[6])


data_path = 'Data_1shot'

save_list_to_file(annealing_time, f'{data_path}/annealing_time.pkl')
save_list_to_file(Ham_time, f'{data_path}/Ham_time.pkl')
save_list_to_file(event_list, f'{data_path}/event_list.pkl')
save_list_to_file(segments, f'{data_path}/segments.pkl')
save_list_to_file(solution_segments, f'{data_path}/solution_segments.pkl')
save_list_to_file(A, f'{data_path}/A.pkl')
save_list_to_file(components, f'{data_path}/components.pkl')
'''
from plotting import *

array_ind = 0
plot_event_tracks(event_list[array_ind])
#plot_event_segments(event_list[array_ind],segments[array_ind])
plot_solution_segments(event_list[array_ind],solution_segments[array_ind])
plot_A(A[array_ind], components[array_ind])
plot_list_vs_index(Ham_time, 'Hamiltonian','BQM')
plot_list_vs_index(annealing_time, 'Annealing','BQM')
'''