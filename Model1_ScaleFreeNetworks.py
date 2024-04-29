import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv
import random
import powerlaw as pl
from Modify_KuramotoModel1 import Kuramoto
from multiprocessing import Pool,cpu_count

random.seed(3407)

def run_simulation(parms):
    N,m = parms
    header = ['coupling', 'r_mean']
    filename = f'OutcomeData/model1_output_{N}.txt'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(' '.join(header))

    G = nx.barabasi_albert_graph(N, m)
    BA_degrees = [G.degree(n) for n in G]
    fit =pl.Fit(BA_degrees,discrete=True)
    gamma = round(fit.power_law.alpha,2)  # gamma约为3
    G_mat = nx.to_numpy_array(G)
    natfreqs = np.random.uniform(-0.5, 0.5, N)
    coupling_vals = np.linspace(0, 1, 50)
    angles_vec = np.random.uniform(-np.pi, np.pi, N)
    runs = []

    for coupling in coupling_vals:
        model = Kuramoto(coupling=coupling, dt=0.1, T=100, n_nodes=N, natfreqs=natfreqs)
        act_mat = model.run(adj_mat=G_mat, angles_vec=angles_vec)
        runs.append(act_mat)

        runs_array = np.array(runs)

    for i, coupling in enumerate(coupling_vals):
        r_mean = np.mean([model.phase_coherence(vec) for vec in runs_array[i, :, -1000:].T])

        # 将结果写入CSV文件
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([N,gamma, coupling, r_mean])

   
    if __name__ == '__main__':
        network_sizes = [(500,3),(1000,3), (2000,3)]
        pool = Pool(processes=cpu_count())
        pool.map(run_simulation, network_sizes)