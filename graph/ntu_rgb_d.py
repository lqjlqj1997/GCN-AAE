import numpy as np

from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]

link = [(1, 2)  ,(2, 21)    ,(3, 21)   ,(4, 3)    ,(5, 21)   ,(6, 5)    , 
        (7, 6)  ,(8, 7)     ,(9, 21)   ,(10, 9)   ,(11, 10)  ,(12, 11)  ,
        (13, 1) ,(14, 13)   ,(15, 14)  ,(16, 15)  ,(17, 1)   ,(18, 17)  ,
        (19, 18),(20, 19)   ,(22, 23)  ,(23, 8)   ,(24, 25)  ,(25, 12)  ]
inward = link
edges = [(i-1,j-1)for (i,j) in link ] + [(j-1,i-1)for (i,j) in link ] + [(i,i) for i in range(25)]


class Graph:
    def __init__(self):
        self.edges = edges
        self.num_nodes = num_node
        
        self.A = tools.get_adjacency_matrix(self.edges, self.num_nodes)

        self.norm_A = tools.normalize_undigraph(self.A)
        self.inward = [(i-1,j-1)for (i,j) in link ]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()

    A, norm_A = graph.A, graph.norm_A
    
    plt.imshow(norm_A, cmap='gray')
    plt.show()
    
