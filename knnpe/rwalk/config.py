import os
import time
import logging
import numpy as np
from multiprocessing import Pool

def wrapper_biased_random_walk(i, ptr, neighs, num_walks, num_steps, p, q, prob_0, prob_1, prob_2):
    walks = []

    for walk in range(num_walks):
        walks_ = []
        curr_pos = i
        prev_pos = i

        walks_.append(i)

        for step in range(num_steps):
            num_neighs = ptr[curr_pos + 1] - ptr[curr_pos]
            if (num_neighs > 0):

                ptr_i = ptr[prev_pos];
                ptr_j = ptr[prev_pos+1];

                while True:
                    if num_neighs == 1:
                        new_pos = neighs[ptr[curr_pos]]
                        break

                    r_r = np.random.uniform()
                    r_step = np.random.randint(0,num_neighs)
                    new_pos = neighs[ptr[curr_pos] + r_step]

                    if ((step==0) or (p==1 and q==1)):
                        break
                    #bias towards returning to the previous node
                    if new_pos == prev_pos:
                        if (r_r < prob_0):
                            break
                    #bias towards moving to the neighborhood of the previous node
                    elif new_pos in neighs[ptr_i:ptr_j]:
                        if (r_r < prob_1):
                            break
                    #bias towards moving to the neighborhood of the previous node
                    elif r_r < prob_2:
                        break
                prev_pos = curr_pos
                curr_pos = new_pos
            walks_.append(curr_pos)
        walks.append(walks_)
    return walks

def biased_random_walk(ptr, neighs, num_walks=10, num_steps=10, p=10, q=0.001, nthreads=-1, seed=-1):
    """
    Executes biased random walks over the nodes of a graph represented in Compressed Sparse Row (CSR) format.

    This function interfaces with a C library (rwalk.so in rwalk.c) to perform efficient 
    computation of the random walks. The signature of the C function is:

    void biased_random_walk(int const* ptr, int const* neighs, int n, int num_walks, 
                            int num_steps, double p, double q, int seed, 
                            int nthreads, int* walks);

    Parameters
    ----------
    ptr : array_1d_int
        Array representing the starting indices of rows in a CSR format graph.
    neighs : array_1d_int
        Array representing the column indices corresponding to non-zero values in a CSR format graph.
    num_walks : int, optional
        The number of walks to start at each node (default is 10).
    num_steps : int, optional
        The number of steps for each walk, including the initial node (default is 10).
    p : float, optional
        Parameter that controls the bias of immediately revisiting a node in the walk (default is 10).
        It is named :math:`{\\lambda}` in the article.
    q : float, optional
        Parameter that controls the bias of moving outside the neighborhood of the previous node (default is 0.001).
        It is named :math:`{\\beta}` in the article.
    nthreads : int, optional
        The number of threads for parallel computation. If -1, uses the maximum number of available 
        threads (default is -1).
    seed : int, optional
        The random seed for walk generation. If -1, a random seed is chosen (default is -1).

    Returns
    -------
    np.ndarray
        An array of shape (n*num_walks, num_steps) containing the node indices of each step in 
        each walk.

    """

    n = ptr.size - 1

    max_prob = max(1.0 / p, max(1.0, 1.0 / q))
    prob_0 = 1.0 / p / max_prob
    prob_1 = 1.0 / max_prob
    prob_2 = 1.0 / q / max_prob
    num_steps = num_steps -1
       
    if seed<0:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    if nthreads<0:
        nthreads = os.cpu_count()
    
    pool = Pool(processes=nthreads)
    args = [(i, ptr, neighs, num_walks, num_steps, p, q, prob_0, prob_1, prob_2) for i in range(n)]
    walks = pool.starmap(wrapper_biased_random_walk, args)
    pool.close()
    pool.join()
        
    return np.vstack(np.asarray(walks))


# def _biased_random_walk(ptr, neighs, num_walks=10, num_steps=10, p=10, q=0.001, nthreads=-1, seed=-1):
#     """
#     Executes biased random walks over the nodes of a graph represented in Compressed Sparse Row (CSR) format.

#     Parameters
#     ----------
#     ptr : array_1d_int
#         Array representing the starting indices of rows in a CSR format graph.
#     neighs : array_1d_int
#         Array representing the column indices corresponding to non-zero values in a CSR format graph.
#     num_walks : int, optional
#         The number of walks to start at each node (default is 10).
#     num_steps : int, optional
#         The number of steps for each walk, including the initial node (default is 10).
#     p : float, optional
#         Parameter that controls the bias of immediately revisiting a node in the walk (default is 10).
#         It is named :math:`{\\lambda}` in the article.
#     q : float, optional
#         Parameter that controls the bias of moving outside the neighborhood of the previous node (default is 0.001).
#         It is named :math:`{\\beta}` in the article.
#     nthreads : int, optional
#         The number of threads for parallel computation. If -1, uses the maximum number of available 
#         threads (default is -1).
#     seed : int, optional
#         The random seed for walk generation. If -1, a random seed is chosen (default is -1).

#     Returns
#     -------
#     np.ndarray
#         An array of shape (n*num_walks, num_steps) containing the node indices of each step in 
#         each walk.

#     """

#     n = ptr.size - 1
#     walks = -np.ones((n*num_walks, num_steps), dtype=np.int32, order='C')
    
#     walks = walks.flatten()
#     max_prob = max(1.0 / p, max(1.0, 1.0 / q))
#     prob_0 = 1.0 / p / max_prob
#     prob_1 = 1.0 / max_prob
#     prob_2 = 1.0 / q / max_prob
#     num_steps = num_steps -1
    
#     if seed<0:
#         import time
#         np.random.seed(int(time.time()))
#     else:
#         np.random.seed(seed)
    
#     for i in range(n):
#         for walk in range(num_walks):

#             curr_pos = i
#             prev_pos = i
#             offset = i * num_walks * (num_steps + 1) + walk * (num_steps + 1);

#             walks[offset] = i

#             for step in range(num_steps):
#                 num_neighs = ptr[curr_pos + 1] - ptr[curr_pos]
#                 if (num_neighs > 0):

#                     ptr_i = ptr[prev_pos];
#                     ptr_j = ptr[prev_pos+1];

#                     while True:
#                         if num_neighs == 1:
#                             new_pos = neighs[ptr[curr_pos]]
#                             break

#                         r_r = np.random.uniform()
#                         r_step = np.random.randint(0,num_neighs)
#                         new_pos = neighs[ptr[curr_pos] + r_step]

#                         if ((step==0) or (p==1 and q==1)):
#                             break
#                         #bias towards returning to the previous node
#                         if new_pos == prev_pos:
#                             if (r_r < prob_0):
#                                 break
#                         #bias towards moving to the neighborhood of the previous node
#                         elif new_pos in neighs[ptr_i:ptr_j]:
#                             if (r_r < prob_1):
#                                 break
#                         #bias towards moving to the neighborhood of the previous node
#                         elif r_r < prob_2:
#                             break
#                     prev_pos = curr_pos
#                     curr_pos = new_pos
#                 walks[offset + step + 1] = curr_pos
    
#     walks = walks.reshape(n*num_walks, num_steps+1)
    
#     return walks
