import numpy as np
import networkx as nx
import os
import csv
import random
from Modify_KuramotoModel2 import Kuramoto

random.seed(3407)

def generate_network(network_size, connections):
    """Generates a Barabasi-Albert network and its adjacency matrix."""
    network = nx.barabasi_albert_graph(network_size, connections)
    return nx.to_numpy_array(network)

def write_results_to_file(network_size, simulation_results, coupling_values, model):
    """Writes the results of simulations to a text file."""
    filename = f'OutcomeData/model2_output_{network_size}.txt'
    with open(filename, 'a') as file:
        for i, coupling in enumerate(coupling_values):
            r_mean = np.mean([model.phase_coherence(vec) for vec in np.array(simulation_results)[i, :, -1000:].T])
            file.write(f"{coupling} {r_mean}\n")


def run_kuramoto_simulation(network_size, connections):
    """Executes the Kuramoto model for a range of coupling values and computes coherence."""
    adjacency_matrix = generate_network(network_size, connections)
    natural_frequencies = np.random.normal(0, 1, network_size)
    coupling_values = np.arange(0, 0.201, 0.02)
    initial_phases = np.random.uniform(-np.pi, np.pi, network_size)
    simulation_results = []
    models = []  # Keep track of models for phase coherence calculation

    for coupling_strength in coupling_values:
        print(coupling_strength)
        model = Kuramoto(coupling=coupling_strength, dt=0.1, T=100, n_nodes=network_size, natfreqs=natural_frequencies)
        activity_matrix = model.run(adj_mat=adjacency_matrix, angles_vec=initial_phases)
        simulation_results.append(activity_matrix)
        models.append(model)  # Store model used for each simulation

    return simulation_results, coupling_values, models[-1]  # Return the last model used

def run_simulation():
    for params in [(500, 2), (1000, 2), (2000, 4)]:
        network_size, connections = params
        print(network_size)
        simulation_results, coupling_values, model = run_kuramoto_simulation(network_size, connections)
        write_results_to_file(network_size, simulation_results, coupling_values, model)

if __name__ == '__main__':
    run_simulation()
