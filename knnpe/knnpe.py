"""
knnpe: A Python package for characterizing unstructured data with the nearest neighbor permutation entropy
==========================================================================================================

``knnpe``: A Python package for characterizing unstructured data with the nearest neighbor permutation entropy [#voltarelli2024]_


If you have used ``knnpe`` in a scientific publication, we would appreciate citations to the following reference [#voltarelli2024]_:

- L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro, 
  `Characterizing unstructured data with the nearest neighbor permutation entropy <https://doi.org/?>`_, 
  ? ?, ? (2024).  `arXiv:? <https://arxiv.org/abs/?>`_

.. code-block:: bibtex
    
   @article{voltarelli2024characterizing,
    title         = {Characterizing unstructured data with the nearest neighbor permutation entropy}, 
    author        = {L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro},
    journal       = {?},
    volume        = {?},
    number        = {?},
    pages         = {?},
    year          = {2024},
    doi           = {},
   }


Released on version 0.1.0 (Marc 2024):

- *k*-nearest neighbor permutation entropy [#voltarelli2024]_

References
==========

.. [#voltarelli2024] L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, 
   R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro. Characterizing unstructured 
   data with the nearest neighbor permutation entropy. ?? 31, 063110 (2024).

For more detailed information about the methods implemented in ``knnpe``, please 
consult its `documentation <https://hvribeiro.github.io/knnpe/_build/html/index.html>`_.

Installing
==========

``knnpe`` can be installed via the command line using

.. code-block:: console

   pip install knnpe

or you can directly clone its git repository:

.. code-block:: console

   git clone https://github.com/hvribeiro/knnpe.git
   cd knnpe
   pip install -e .


Basic usage
===========

We provide a `notebook <https://github.com/hvribeiro/knnpe/blob/master/examples/knnpe.ipynb>`_
illustrating how to use ``knnpe``. The code below shows simple applications of ``knnpe``.

.. code-block:: python


.. figure:: https://raw.githubusercontent.com/?.png
   :height: 489px
   :width: 633px
   :scale: 80 %
   :align: center


Contributing
============

Pull requests addressing errors or adding new functionalities are always welcome.

List of functions
=================

"""

import numpy as np

from math                 import factorial
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors    import kneighbors_graph
from tqdm                 import tqdm

from rwalk import *


def knn_permutation_entropy(data, d=3, tau=1, p=10, q=0.001, random_walk_steps=10, 
                            num_walks=10, n_neighbors=25, nthreads=-1, hide_bar=True, metrics=False):
    """
    Estimates the k-nearest neighbor permutation entropy of unstructured data
    
    Parameters
    ----------
    data : ndarray
        Input array containing unstructured data points, where each row is in the form [x, y, value].
    d : int, optional
        The embedding dimension for the entropy calculation (default is 3).
    tau : int, optional
        The embedding delay for the entropy calculation (default is 1).
    p : float, optional
        Parameter that controls the bias of immediately revisiting a node in the walk (default is 10).
        It is named :math:`{\\lambda}` in the article.
    q : float, optional
        Parameter that controls the bias of moving outside the neighborhood of the previous node (default is 0.001).
        It is named :math:`{\\beta}` in the article.
    random_walk_steps : int, optional
        The number of steps in each random walk (default is 10).
    num_walks : int, optional
        The number of random walk samples to start from each node (default is 10).
    n_neighbors : int or array-like, optional
        The number of neighbors for constructing the k-nearest neighbor graph (default is 25).
    nthreads : int, optional
        The number of parallel threads for the computation (default is -1, which uses all available cores).
    hide_bar : bool, optional
        If True, the progress bar is not displayed (default is False).
    metrics : bool, optional
        If True, calculates graph density and largest component fraction (default is False).
    Returns
    -------
    float or np.ndarray
       The knn permutation entropy by default or an array of values for n_neighbors.
           If metrics is True, for each value in n_neighbors, it returns: 
               [n_neighbors, graph largest component fraction, graph density, knn entropy]
    """


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def estimate_entropy_from_walks(time_series, all_walks, d, tau):
        """
        Estimates the permutation entropy of a time series based on a set of random walks.


        Parameters
        ----------
        time_series : ndarray
            The time series data as a 1D numpy array. Each element represents a data point.
        all_walks : list of ndarrays
            A list where each element is a numpy array representing a random walk. Each array
            contains indices corresponding to the time points in the time series.
        d : int
            The embedding dimension for the entropy calculation.
        tau : int
            The embedding delay for the entropy calculation.

        Returns
        -------
        float
            The knn permutation entropy of the time series based on the provided random walks.
        """
    
        ordinal_sequence = []

        for walk in all_walks:
            ts = np.expand_dims(time_series[walk],axis=0)

            partitions = np.apply_along_axis(func1d       = np.lib.stride_tricks.sliding_window_view, 
                                             axis         = 1, 
                                             arr          = ts, 
                                             window_shape = (d+(d-1)*(tau-1),)
                                             )[::, ::, ::tau].reshape(-1, d)

            ordinal_sequence += [np.apply_along_axis(np.argsort, 1, partitions)]

        ordinal_sequence = np.vstack(ordinal_sequence)
        _, symbols_count = np.unique(ordinal_sequence, return_counts=True, axis=0)
        probabilities    = symbols_count/symbols_count.sum()

        S = -np.sum(probabilities*np.log(probabilities))
        
        return S
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



    data = np.asarray(data)

    data_to_graph = data[:,0:-1]
    time_series = data[:,-1]

    if not isinstance(n_neighbors, (list,np.ndarray)):
        n_neighbors = [n_neighbors]

    number_of_vertex = len(time_series)
    max_number_of_egdes = number_of_vertex*(number_of_vertex-1)/2.
    
    Smax  = np.log(factorial(d))

    property_list = []
    
    for n_neighbor in tqdm(n_neighbors, position=0, leave=True, disable=hide_bar):
        
        adj_matrix = kneighbors_graph(data_to_graph, n_neighbors=n_neighbor, n_jobs=-1)
        adj_matrix = ((adj_matrix+adj_matrix.T)>0).astype(int)
        adj_matrix_csr = adj_matrix.tocsr()
        
        ptr, neighs = adj_matrix_csr.indptr, adj_matrix_csr.indices

        all_walks = biased_random_walk(ptr, neighs, num_steps=random_walk_steps, 
                                       num_walks=num_walks, p=p, q=q, 
                                       nthreads=nthreads, seed=np.random.randint(1,1e6))

        S = estimate_entropy_from_walks(time_series, all_walks, d, tau)/Smax
        
        if metrics:
            ncomonents, labels = connected_components(adj_matrix)
            component_ids, nvertex_component = np.unique(labels,return_counts=True)
            frac_largest_component = max(nvertex_component)/number_of_vertex
            number_of_egdes = int(np.sum(adj_matrix)/2)
            graph_density = number_of_egdes/max_number_of_egdes
            property_list.append(np.asarray([n_neighbor, frac_largest_component, graph_density, S]))
        else:
            property_list.append(S)

    if len(n_neighbors)>1:
        property_list = np.vstack(property_list)
    else:
        property_list = property_list[0]

    return property_list
