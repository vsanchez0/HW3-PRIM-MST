import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None
        self.V=len(self.adj_mat)

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        
        key = [float('inf')] * self.V   # key value to pick min edge weight
        parent = [None] * self.V
        key[0] = 0                      # first vertex
        mstSet = [False] * self.V       # initialize empty set

        parent[0] = -1

        for _ in range(self.V):
            # choose min distrance vertex
            u = self.minKey(key, mstSet)

            mstSet[u] = True

            for v in range(self.V):
                # update key if graph[u][v] < key[v]
                if self.adj_mat[u][v] > 0 and mstSet[v] == False \
                and key[v] > self.adj_mat[u][v]:
                    key[v] = self.adj_mat[u][v]
                    parent[v] = u

        self.mst = self._create_mst_adjacency_matrix(parent)

    def _create_mst_adjacency_matrix(self, parent: list) -> np.ndarray:
        """
        Generate the adjacency matrix for the MST based on the parent array.
        """
        adjacency_matrix = np.zeros((self.V, self.V), dtype=float)
        for i in range(1, self.V):
            u = parent[i]
            v = i
            weight = self.adj_mat[i][u]
            adjacency_matrix[u][v] = weight
            adjacency_matrix[v][u] = weight
        return adjacency_matrix

    def minKey(self, key, mstSet):
        """
        Pick the vertex with the minimum key value from the set of vertices not yet included in MST.
        """
        min = float('inf')

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index