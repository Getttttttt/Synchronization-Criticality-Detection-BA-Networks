import numpy as np
import networkx as nx
import csv
import random
import powerlaw as pl
from Modify_KuramotoModel1 import Kuramoto
from multiprocessing import Pool, cpu_count

random.seed(3407)

def create_barabasi_albert_network(size, connections):
    """Generates a Barabasi-Albert graph."""
    return nx.barabasi_albert_graph(size, connections)

def calculate_power_law_exponent(degrees):
    """Calculates the power law exponent using the powerlaw library."""
    fit = pl.Fit(degrees, discrete=True)
    return round(fit.power_law.alpha, 2)  # Typical value approximates to 3

def simulate_kuramoto_model(network_size, connections):
    """Runs the Kuramoto model simulation and writes results to a file."""
    header = ['coupling', 'r_mean']
    filename = f'OutcomeData/model1_output_{network_size}.txt'

    with open(filename, 'w', newline='') as file:
        file.write(' '.join(header))

    network = create_barabasi_albert_network(network_size, connections)
    degree_sequence = [network.degree(node) for node in network]
    gamma = calculate_power_law_exponent(degree_sequence)
    adjacency_matrix = nx.to_numpy_array(network)
    natural_frequencies = np.random.uniform(-0.5, 0.5, network_size)
    coupling_values = np.linspace(0, 1, 50)
    initial_phases = np.random.uniform(-np.pi, np.pi, network_size)
    simulation_results = []

    for coupling_strength in coupling_values:
        model = Kuramoto(coupling=coupling_strength, dt=0.1, T=100, n_nodes=network_size, natfreqs=natural_frequencies)
        activity_matrix = model.run(adj_mat=adjacency_matrix, angles_vec=initial_phases)
        simulation_results.append(activity_matrix)

        results_array = np.array(simulation_results)

    for index, coupling_strength in enumerate(coupling_values):
        coherence_mean = np.mean([model.phase_coherence(phase_vec) for phase_vec in results_array[index, :, -1000:].T])

        # Write results to file
        with open(filename, 'a') as file:
            file.write(f"{coupling_strength} {coherence_mean}\n")

if __name__ == '__main__':
    network_configs = [(500, 3), (1000, 3), (2000, 3)]
    with Pool(processes=cpu_count()) as pool:
        pool.map(simulate_kuramoto_model, network_configs)
