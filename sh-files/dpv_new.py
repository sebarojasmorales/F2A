import pandas as pd
import numpy as np 
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def read_asc(file_path):
    with open(file_path, 'r') as file:
        # Leer las primeras seis líneas y almacenarlas
        header = [next(file) for _ in range(6)]
        # Leer los datos numéricos
        data = np.loadtxt(file, dtype=np.float32)
    return header, data

def write_asc(file_path, header, array):
    with open(file_path, 'w') as file:
        # Escribir el encabezado
        file.writelines(header)
        # Escribir los datos numéricos
        np.savetxt(file, array, fmt='%.6f')

# forest_layer = "/home/sebar/Documents/SAESA/Datos/fuels.asc"
values_risk_file = "/home/sebar/Downloads/values_risk.asc"
header, values_risk = read_asc(values_risk_file)
shape = values_risk.shape
ncells = values_risk.size
values_risk = values_risk.reshape([-1])

n_threads = 10
nsims = 5

archivos = [f"/home/sebar/Documents/SAESA/Pruebas DPV/5Messages/MessagesFile{str(i).zfill(4)}.csv" for i in range(1,nsims+1)]
blocks = np.array_split(archivos, n_threads)

#def calculate_value(archives):
#    final = np.zeros(ncells)
#    for archive in archives:
#        message = pd.read_csv(archive, header=None)
#        graph = nx.DiGraph()
#        for _, row in message.iterrows():
#            graph.add_edge(int(row[0])-1, int(row[1])-1, weight=row[2])

#        nodes = list(graph.nodes)
#        dpv = lambda node: values_risk[list(nx.descendants(graph, node))].sum()
#        values = np.zeros(ncells)
#        values[nodes] = values_risk[nodes] + np.array(list(map(dpv, nodes)))
#        final = final + values
#    return final

def calculate_value(archives):
    final = np.zeros(ncells)
    for archive in archives:
        message = pd.read_csv(archive, header=None)
        graph = nx.DiGraph()
        for _, row in message.iterrows():
            graph.add_edge(int(row[0])-1, int(row[1])-1, weight=row[2])

        root = int(message[0][0])-1
        shortest_paths = nx.single_source_dijkstra_path(graph, root, weight='weight')

        # MEJORAR
        new_graph = nx.DiGraph()
        for destino, camino in shortest_paths.items():
            # Agregar las aristas del camino
            for i in range(len(camino) - 1):
                u, v = camino[i], camino[i + 1]
                new_graph.add_edge(u, v)

        nodes = list(new_graph.nodes)
        dpv = lambda node: values_risk[list(nx.descendants(new_graph, node))].sum()
        values = np.zeros(ncells)
        values[nodes] = values_risk[nodes] + np.array(list(map(dpv, nodes)))
        final = final + values
    return final

inicio = time.time()

with ProcessPoolExecutor(max_workers=n_threads) as executor: 
    resultados = executor.map(calculate_value, blocks)

dpv_final = np.zeros(ncells)
for dpv_partial in resultados:
    dpv_final+=dpv_partial
dpv_final = dpv_final.reshape(shape)

write_asc('/home/sebar/Downloads/dpv.asc', header, dpv_final)

fin = time.time()
print(fin - inicio)

