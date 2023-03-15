# prerequisites
import copy
import os

import numpy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from networkx import erdos_renyi_graph

"""
plot a network
"""
def plotNetwork(graph, communities = [1, 1, 1, 1, 1, 1]):
    np.random.seed(123) #to freeze the graph's view (networks uses a random view)
   # A=np.matrix(network["mat"])
    #G=nx.from_numpy_matrix(A)
    pos = nx.spring_layout(graph)  # compute graph layout
    plt.figure(figsize=(4, 4))  # image is 8 x 8 inches
    nx.draw_networkx_nodes(graph, pos, node_size=50, cmap=plt.cm.RdYlBu, node_color = communities)
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    plt.show()


'''
Verifica daca exista conexiune intre doua comunitati
'''
def connectionExists(com1, com2, comunitati, graph):
     #caut toate nodurile vecine comunitatii 1
    noduri_vecine = []
    for nod in com1:
        listaAdiacenta = graph.adj[nod]
        for nod in listaAdiacenta:
            if (nod not in noduri_vecine):
                noduri_vecine.append(nod)

    #vad daca vreo unul din ele este din comunitatea 2
    for nod in noduri_vecine:
        if nod in com2:
            return True
    return False

'''
Calculez modularitatea
'''
def calculateQ(graph, comunitati, comi, comj):
    numarMuchiiTotal = len(graph.edges)
    q = 0
    for com in comunitati:
        if (com != comi and com != comj):
            sumaGradelorNodurilor = 0
            numarMuchiiComunitate = 0
            for nod in com:
                sumaGradelorNodurilor += graph.degree[nod]

            for nodi in com:
                for nodj in com:
                    if nodi in graph.adj[nodj]:
                        numarMuchiiComunitate += 1

            numarMuchiiComunitate /= 2 #cause i end up countiing them twice

            q += (numarMuchiiComunitate / numarMuchiiTotal) - (sumaGradelorNodurilor * sumaGradelorNodurilor / (4 * numarMuchiiTotal * numarMuchiiTotal))

    sumaGradelorNodurilor = 0
    numarMuchiiComunitate = 0
    for nod in comi:
        sumaGradelorNodurilor += graph.degree[nod]
    for nod in comj:
        sumaGradelorNodurilor += graph.degree[nod]

    comi.extend(comj)

    numarMuchiiComunitate1 = 0
    for nodi in comi:
        for nodj in comi:
            if nodi in graph.adj[nodj]:
                numarMuchiiComunitate1 += 1

    numarMuchiiComunitate1/=2

    q += (numarMuchiiComunitate1 / numarMuchiiTotal) - (sumaGradelorNodurilor * sumaGradelorNodurilor / (4 * numarMuchiiTotal * numarMuchiiTotal))

    return q

'''
Detec comunities with greedy and Q
'''
def detec_comunitati():

    #EXAMPLE
    graph = nx.read_gml("communityDetection/real/dolphins/dolphins.gml",label='id')
    #graph = nx.read_gml("communityDetection/real/football/football.gml",label='id')
    #graph = nx.read_gml("communityDetection/real/karate/karate.gml",label='id')
    #graph = nx.read_gml("communityDetection/real/krebs/krebs.gml",label='id')

    #graph = nx.read_gml("communityDetection/generated/graph1.gml",label='id')
    #graph = nx.read_gml("communityDetection/generated/graph3.gml",label='id')
    #graph = nx.read_gml("communityDetection/generated/graph4.gml",label='id')
    #graph = nx.read_gml("communityDetection/generated/graph6.gml",label='id')
    #graph = nx.read_gml("communityDetection/generated/graph9.gml",label='id')
    #graph = nx.read_gml("communityDetection/generated/graph11.gml",label='id')

    comunitati = []  #initially all nods are separate comunities
    for nod in graph:
        comunitati.append([nod])

    optimal_network = []  #the comunities in the end
    qMax = -1000          #maxima valoare a lui Q
    comunitiesLeftToMerge = len(comunitati)

    #se opreste cand toate nodurile formeaza o comunitate
    while(comunitiesLeftToMerge > 1):
        qMaxLocal = -1000
        com2tomerge = []
        com1tomerge = []

        visited = []
        #iau pe rand doua comunitati
        #daca nu le-am vizitat innainte si exista legatura intre ele
        #calculez Q pe reteaua imaginara unde cele doua comunitati ar fi legate
        #daca q e maximal, il salvez pe el si reteaua cu cele 2 com legate
        for comi in comunitati:
            visited.append(comi)
            for comj in comunitati:
                if comj not in visited:
                    if connectionExists(comi, comj, comunitati, graph):
                        q = calculateQ(graph, comunitati, comi.copy(), comj.copy())
                        if q >= qMaxLocal:
                            qMaxLocal = q
                            com1tomerge=comi
                            com2tomerge=comj


        print(qMaxLocal)

        #leg cele doua comunitati care corespun Q-ului maximal
        com1tomerge.extend(com2tomerge)
        comunitati.remove(com2tomerge)

        #salvez noul Q maximal si reteaua
        if (qMaxLocal > qMax):
            qMax=qMaxLocal
            optimal_network = comunitati.copy()
        comunitiesLeftToMerge = len(comunitati)
        print(comunitiesLeftToMerge)


    #trasnform reteaua gasita intr-un format ok pentru functia plotNetwork
    comunitatiDictionar = {}
    indexComunitate = 1

    print("Comunitati sub forma lista:")
    print(optimal_network)

    for com in optimal_network:
        for nod in com:
            comunitatiDictionar[nod] = indexComunitate
        indexComunitate += 1

    print("Comunitati sub forma de dictionar {nod : comunitatea lui}")
    print(comunitatiDictionar)
    dictionarToLista = []
    for nod, val in comunitatiDictionar.items():
        dictionarToLista.append(val)

    print("Comunitati individuale:")
    print(numpy.unique(dictionarToLista))

    #afisez reteaua finala
    plotNetwork(graph, dictionarToLista)


#PROGRAM START
detec_comunitati()

"""
#graph generation
g = erdos_renyi_graph(n=60, p=0.2)
nx.write_gml(g, 'communityDetection/generated/graph6.gml')"""


'''
#old code
        for i in range(0, len(comunitati)-2):
            #print(i)
            for j in range(i+1, len(comunitati)-1):
                #print(j)
                if (connectionExists(i, j, comunitati, graph)):
                    #print("yes")
                    #tempNetwork = connectComunitati(i, j, cpy)
                    q = calculateQ(graph, comunitati, i, j) # calc q if i connected i and j
                    if (q > qMaxLocal):
                        #print("better q")
                        #print(tempNetwork)
                        #save q and the network
                        qMaxLocal = q
                        com1tomerge=comunitati[i]
                        com2tomerge=comunitati[j]
                        #optimalNetworksLocal = comunitati
    '''