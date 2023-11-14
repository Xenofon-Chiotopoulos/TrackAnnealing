import pickle
import matplotlib.pyplot as plt

def plot_list_vs_index(T, title='Annealing'):
    plt.figure(figsize=(10, 6))  
    plt.plot(range(1, len(T) + 1), T, marker='o', linestyle='-', color='b', label=f'{title} Time(s)')
    plt.xlabel('Number of Tracks', fontsize=14)
    plt.ylabel(f'{title} Time(s)', fontsize=14)
    plt.title(f'Plot of {title} Time(s) vs Number of Tracks', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout() 
    plt.show()

def load_list_from_file(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

Ham_time = load_list_from_file('Data/Ham_time.pkl')
annealing_time = load_list_from_file('Data/annealing_time.pkl')

plot_list_vs_index(Ham_time, 'Hamiltonian')
plot_list_vs_index(annealing_time, 'Annealing')
