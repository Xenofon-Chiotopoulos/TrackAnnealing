import pickle
from plotting import *

def load_list_from_file(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

data_path = 'Data_1shot'

Ham_time = load_list_from_file(f'{data_path}/Ham_time.pkl')
annealing_time = load_list_from_file(f'{data_path}/annealing_time.pkl')
event = load_list_from_file(f'{data_path}/event_list.pkl')
A = load_list_from_file(f'{data_path}/A.pkl')
components = load_list_from_file(f'{data_path}/components.pkl')
segments = load_list_from_file(f'{data_path}/segments.pkl')
solution_segments = load_list_from_file(f'{data_path}/solution_segments.pkl')

array_ind = -1
''''''
plot_event_tracks(event[array_ind])
plot_event_segments(event[array_ind],segments[array_ind])
plot_solution_segments(event[array_ind],solution_segments[array_ind])
plot_A(A[array_ind], components[array_ind])


plot_list_vs_index(Ham_time, 'Hamiltonian','BQM', True)
plot_list_vs_index(annealing_time, 'Annealing','BQM', True)


def evaluate_solution(event, solution_segments):
    correct_segments = 0
    total_segments = 0
    correct_per_track = {}

    for track in event.tracks:
        correct_per_track[track.track_id] = 0

    for solution_segment in solution_segments:
        for track in event.tracks:
            total_segments += 1
            if (solution_segment.from_hit in track.hits) and (solution_segment.to_hit in track.hits):
                correct_segments += 1
                correct_per_track[track.track_id] += 1
    #print(total_segments)
    #print(correct_segments)
    return correct_segments, correct_per_track

correct_segments, correct_per_track = evaluate_solution(event[array_ind], solution_segments[array_ind])
print(f"Total Correct Segments: {correct_segments}")

for track_id, count in correct_per_track.items():
    print(f"Track {track_id}: {count} Correct Segments")

