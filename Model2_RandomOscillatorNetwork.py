import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import csv
import random
from Modify_KuramotoModel2 import Kuramoto

random.seed(3407)


# 循环不同的网络大小
# N=500,m=8,gamma=2.79
# N=1000,m=8,gamma=2.79
# N=2000,m=7,gamma=2.79
# N=4000,m=5,gamma=2.8
# N=5000,m=7,gamma=2.81
for prams in [(500,8),(2000,7),(4000,5),(5000,7)]:
    N,m = prams
    print(N)
    G = nx.barabasi_albert_graph(N, m)
    G_mat = nx.to_numpy_array(G)
    natfreqs = np.random.normal(0, 1, N)
    coupling_vals = np.arange(0, 0.201, 0.02)
    angles_vec = np.random.uniform(-np.pi, np.pi, N)
    runs = []

    for coupling in coupling_vals:
        print(coupling)
        model = Kuramoto(coupling=coupling, dt=0.1, T=100, n_nodes=N, natfreqs=natfreqs)
        act_mat = model.run(adj_mat=G_mat, angles_vec=angles_vec)
        runs.append(act_mat)

    runs_array = np.array(runs)
    plt.figure()

    for i, coupling in enumerate(coupling_vals):
        r_mean = np.mean([model.phase_coherence(vec) for vec in runs_array[i, :, -1000:].T])
        plt.scatter(coupling, r_mean, c='steelblue', s=20, alpha=0.7)

        # 将结果写入CSV文件
        with open('data/results2.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([N, coupling, r_mean])

    # Kc = 2 / (np.pi * 1 * N)
    # plt.vlines(Kc, 0, 1, linestyles='--', color='orange')
    # r_theory = np.sqrt(coupling_vals / N - Kc)
    # plt.plot(coupling_vals, r_theory, label=r'$r = \sqrt{K-K_c}$')

    # plt.legend()
    plt.grid(linestyle='--', alpha=0.8)
    plt.ylabel('order parameter (R)')
    plt.xlabel(r'$\lambda$')

    # 保存图形
    plt.savefig(f'fig3/N_{N}.png')
    plt.close()
