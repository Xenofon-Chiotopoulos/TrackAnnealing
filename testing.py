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

N_MODULES = 3
N_TRACKS = 2

Ham_time = []
annealing_time = []
for NTracks in range(1,10):

    LX = 2
    LY = 2
    SPACING = 1

    detector = toy.generate_simple_detector(N_MODULES, LX, LY, SPACING)
    event = toy.generate_event(detector, NTracks, theta_max=np.pi / 50, seed=1)

    a, b = simulated_annealing(event, params)
    Ham_time.append(a)
    annealing_time.append(b)

save_list_to_file(annealing_time, 'Data/annealing_time.pkl')
save_list_to_file(Ham_time, 'Data/Ham_time.pkl')
