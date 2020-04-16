import numpy as np

def get_adjacency_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i,j in edges:
        A[i,j] = 1.
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape    
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD






# def get_hop_distance(num_node, edge, max_hop=1):
#     A = np.zeros((num_node, num_node))
#     for i, j in edge:
#         A[j, i] = 1
#         A[i, j] = 1

#     # compute hop steps
#     hop_dis = np.zeros((num_node, num_node)) + np.inf
#     transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
#     arrive_mat = (np.stack(transfer_mat) > 0)
#     for d in range(max_hop, -1, -1):
#         hop_dis[arrive_mat[d]] = d
#     return hop_dis


# def normalize_adjacency_matrix(A):
#     node_degrees = A.sum(-1)
#     degs_inv_sqrt = np.power(node_degrees, -0.5)
#     norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
#     return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


# def k_adjacency(A, k, with_self=False, self_factor=1):
#     assert isinstance(A, np.ndarray)
#     I = np.eye(len(A), dtype=A.dtype)
#     if k == 0:
#         return I
#     Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
#        - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
#     if with_self:
#         Ak += (self_factor * I)
#     return Ak