import matplotlib.pyplot as plt
import numpy as np

def plot_list_vs_index(T, title='Annealing', annealing_type='BQM', eqn=False):
    plt.figure(figsize=(10, 6))  
    rng = np.array(range(1, len(T) + 1))
    plt.plot(range(1, len(T) + 1), np.array(T), marker='o', linestyle='-', color='b', label=f'{title} Time(s)')
    plt.xlabel('Number of Tracks', fontsize=14)
    plt.ylabel(f'{title} Time(s)', fontsize=14)
    plt.title(f'Plot of {annealing_type} {title} Time(s) vs Number of Tracks', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    #plt.yscale('log')
    plt.tight_layout() 
    plt.show()

    if eqn == True:
        x = np.array(range(1, len(T) + 1))
        y = np.array(T)

        coefficients = np.polyfit(x, np.log(y), 1)
        a, b = coefficients
        equation_of_curve = f'y = {np.exp(b):.4f} * e^({a:.4f}x)'
        #plt.figure(figsize=(8,6),dpi=450)
        plt.plot(x, np.exp(b) * np.exp(a * x), color='r', label=f'Fitted Exponential Curve: {equation_of_curve}')
        plt.xlabel('Number of Tracks')
        plt.title('Annealing Time Scaling')
        plt.ylabel('Time(s)')
        plt.grid()
        plt.legend()
        plt.show()
        return equation_of_curve

def plot_event_tracks(event):
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = plt.axes(projection='3d')
    event.display(ax)
    ax.view_init(vertical_axis='y')
    fig.set_tight_layout(True)
    ax.axis('off')
    ax.set_title(f"Generated event\n{len(event.modules)} modules\n{len(event.tracks)} tracks - {len(event.hits)} hits")
    plt.show()

def plot_event_segments(event,segments):
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

def plot_solution_segments(event,solution_segments,segments):
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = plt.axes(projection='3d')
    for evnt in event:
        evnt.display(ax, show_tracks=False)

    for segment in solution_segments:
        segment.display(ax)

    ax.set_title(f"{len(np.array(segments).flatten())} segments generated, {len(np.array(solution_segments).flatten())/3} Tracks found" )
    ax.view_init(vertical_axis='y')
    fig.set_tight_layout(True)
    ax.axis('off')
    plt.show()

def plot_A(A, components):
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