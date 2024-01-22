import numpy as np

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph():
    num_node = 30
    self_link = [(i, i) for i in range(num_node)]
    inward_ori_index = [(2, 15), (5, 4), (4,3), (3, 15), (8,7), (7, 6), (6, 15),(15, 1), (11, 10), (10, 9), (9, 1), (14, 13), (13, 12),(12, 1),
                        (17, 30), (20, 19), (19,18), (18, 30), (23,22), (22, 21), (21, 30),(30, 16), (26, 25), (25, 24), (24, 30), (29, 28), (28, 27),(27, 30)]
    # inward_ori_index = [(3, 16), (16, 2), (1,2), (6, 5), (5,4), (4, 16), (9, 8),
    #                     (8, 7), (7, 16), (12, 11), (11, 10), (10, 1), (15, 14),
    #                     (14, 13),(13,1),(19,32),(32,18),(17,18),(22,21),(21,20),(20,32),(25,24),(24,23),(23,32),(28,27),(27,26),(26,17),(31,30),(30,29),(29,17)]
    inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
    outward = [(j, i) for (i, j) in inward]
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A