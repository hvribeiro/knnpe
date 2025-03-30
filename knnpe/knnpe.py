"""
knnpe: A Python package implementing the *k*-nearest neighbor permutation entropy
=================================================================================

The *k*-nearest neighbor permutation entropy [#voltarelli2024]_ extends the fundamental premise of investigating 
the relative ordering of time series elements [#bandtpompe2002]_ or image pixels [#ribeiro2012]_ inaugurated by 
permutation entropy to unstructured datasets. This method builds upon nearest neighbor graphs to establish neighborhood
relations among data points and uses random walks over these graphs to extract ordinal patterns and their distribution, 
thereby defining the $k$-nearest neighbor permutation entropy.

.. note::
  
   If you have used ``knnpe`` in a scientific publication, we would appreciate citations to the following reference:

   - L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro, 
     `Characterizing unstructured data with the nearest neighbor permutation entropy <https://doi.org/10.1063/5.0209206>`_, 
     Chaos 34, 053130 (2024). `arXiv:2403.13122 <https://arxiv.org/abs/2403.13122>`_

   .. code-block:: bibtex
       
      @article{voltarelli2024characterizing,
       title         = {Characterizing unstructured data with the nearest neighbor permutation entropy}, 
       author        = {L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro},
       journal       = {Chaos},
       volume        = {34},
       pages         = {053130},
       year          = {2024},
       doi           = {10.1063/5.0209206},
      }

Installing
==========

``knnpe`` uses `OpenMP <https://www.openmp.org/>`_ and `GNU Scientific Library (GSL) <https://www.gnu.org/software/gsl/>`_ 
to implement a parallelized numerically efficient random walk function. This function is written in C and it is integrated with our 
Python module via the `ctypes <https://docs.python.org/3/library/ctypes.html>`_ library. To use this function, you must have OpenMP and GSL installed before installing ``knnpe``. 

In Ubuntu/Debian, you can install these dependencies via apt:

.. code-block:: console

   sudo apt install build-essential
   sudo apt install libgsl-dev
   sudo apt install libomp-dev

If these dependencies are not available, ``knnpe`` will use a native Python function to do the random walks. This function is also parallelized and may work nicely for most applications; still, it is significantly slower than its C counterpart. For large datasets, we strongly recommend using the C version.

If all dependencies are available, ``knnpe`` can be installed via:

.. code-block:: console

   pip install git+https://github.com/hvribeiro/knnpe

or

.. code-block:: console

   git clone https://github.com/hvribeiro/knnpe.git
   cd knnpe
   pip install -e .

If all dependencies are **not** available, you can use the PyPI version via:

.. code-block:: console

   pip install knnpe

Basic usage
===========
Implementation of the $k$-nearest neighbor permutation entropy. (A) Illustration of a dataset with irregularly distributed data points $\\{z_i\\}_{i=1,\\dots,N}$ in the $xy$-plane where each coordinate pair $(x_i,y_i)$ is associated with a value $z_i$. (B) Initially, we construct a $k$-nearest neighbor graph using the data coordinates to define neighborhood relationships. In this graph, each data point $z_i$ represents a node, with undirected edges connecting pairs $i\\leftrightarrow j$ when $j$ is among the $k$-nearest neighbors of $i$ ($k=3$ in this example). (C) Subsequently, we execute $n$ biased random walks of length $w$ starting from each node, sampling the data points to generate time series ($n=2$ and $w=6$ in this example). We then apply the Bandt-Pompe approach to each of these time series. This involves creating overlapping partitions of length $d$ (embedding dimension) and arranging the partition indices in ascending order of their values to determine the sorting permutations for each partition ($d=3$ in this example). (D) Finally, we evaluate the probability of each of the $d!$ possible permutations (ordinal distribution) and calculate its Shannon entropy, thereby defining the $k$-nearest neighbor permutation entropy.

.. figure:: https://raw.githubusercontent.com/hvribeiro/knnpe/main/examples/figs/figmethod.png
   :scale: 80 %
   :align: center

The function `knn_permutation_entropy` of ``knnpe`` calculates $k$-nearest neighbor permutation entropy as illustrated below for a random dataset with three columns.

.. code-block:: python

   import numpy as np
   from knnpe import knn_permutation_entropy
   
   data = np.random.normal(size=(100,3))
   knn_permutation_entropy(data)

The last column in `data` corresponds to $\\{z_i\\}_{i=1,\\dots,N}$ values, while the first two columns are used as the data coordinates $\\vec{r}_i = (x_i,y_i)$. If the dataset has more dimensions in data coordinates, they must be passed as the first columns of the dataset, and the last column is always assumed to correspond to $z_i$ values. The code below illustrates the case of data with three dimensions in data coordinates:

.. code-block:: python

   import numpy as np
   from knnpe import knn_permutation_entropy
   
   data = np.random.normal(size=(100,4))
   knn_permutation_entropy(data)

The function `knn_permutation_entropy` has the following parameters:

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
 complexity : bool, optional
     If True, also calculates the knn permutation complexity.
 dis_metric : string, optional
     The distance metric used to determine the knn graph (default is 'euclidean'). It should be an
     string corresponding to one sklearn.metrics.DistanceMetric.

We provide a `notebook <https://github.com/hvribeiro/knnpe/blob/main/examples/knnpe.ipynb>`_
illustrating how to use ``knnpe`` and further information we refer to the knnpe's `documentation <https://hvribeiro.github.io/knnpe/>`_

Contributing
============

Pull requests addressing errors or adding new functionalities are always welcome.

References
==========

.. [#voltarelli2024] L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro. 
   Characterizing unstructured data with the nearest neighbor permutation entropy. 
   Chaos 34, 053130 (2024).

.. [#bandtpompe2002] C. Bandt, B. Pompe. 
   Permutation entropy: A Natural Complexity Measure for Time Series. 
   Physical Review Letters 88, 174102 (2002).

.. [#ribeiro2012] H. V. Ribeiro, L. Zunino, E. K. Lenzi, P. A. Santoro, R. S. Mendes.
   Complexity-Entropy Causality Plane as a Complexity Measure for Two-Dimensional Patterns. 
   PLOS ONE 7, e40689 (2012).

"""

import numpy as np

from math                 import factorial
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors    import kneighbors_graph
from tqdm                 import tqdm

from .rwalk import *


def knn_permutation_entropy(data, d=3, tau=1, p=10, q=0.001, random_walk_steps=10, 
                            num_walks=10, n_neighbors=25, nthreads=-1, hide_bar=True, metrics=False,
                            complexity=False, dis_metric='euclidean'):
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
    complexity : bool, optional
        If True, also calculates the knn permutation complexity.
    dis_metric : string, optional
        The distance metric used to determine the knn graph (default is 'euclidean').
        It should be a string corresponding to one sklearn.metrics.DistanceMetric.
        If 'mahalanobis', it will calculate the inverve covariance matrix automatically.
        
    Returns
    -------
    float or np.ndarray
       The knn permutation entropy by default or an array of values for n_neighbors.
           If metrics is True, for each value in n_neighbors, it returns: 
               [n_neighbors, graph largest component fraction, graph density, knn entropy]
        If complexity=True, knn entropy is replace by a list: [knn entropy, knn permutation complexity].
    """


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def estimate_entropy_from_walks(time_series, all_walks, d, tau, complexity=False):
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
        complexity : bool, optional
            If True, also calculates the permutation complexity.

        Returns
        -------
        float
            The knn permutation entropy of the time series based on the provided random walks.
            If complexity=True, knn entropy is replace by list: [knn entropy, knn permutation complexity].
        """

        ordinal_sequence = []

        Smax  = np.log(factorial(d))

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

        S = -np.sum(probabilities*np.log(probabilities))/Smax

        if complexity:
            n = float(factorial(d))
            n_states_not_occuring = n-len(probabilities)
            uniform_dist          = 1/n

            p_plus_u_over_2      = (uniform_dist + probabilities)/2  
            s_of_p_plus_u_over_2 = -np.sum(p_plus_u_over_2*np.log(p_plus_u_over_2)) - (0.5*uniform_dist)*np.log(0.5*uniform_dist)*n_states_not_occuring

            s_of_p_over_2 = -np.sum(probabilities*np.log(probabilities))/2
            s_of_u_over_2 = np.log(n)/2.

            js_div_max = -0.5*(((n+1)/n)*np.log(n+1) + np.log(n) - 2*np.log(2*n))    
            js_div     = s_of_p_plus_u_over_2 - s_of_p_over_2 - s_of_u_over_2

            return [S, S*js_div/js_div_max]
        
        return S
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



    data = np.asarray(data)

    data_to_graph = data[:,0:-1]
    time_series = data[:,-1]

    if not isinstance(n_neighbors, (list,np.ndarray)):
        n_neighbors = [n_neighbors]

    number_of_vertex = len(time_series)
    max_number_of_egdes = number_of_vertex*(number_of_vertex-1)/2.

    property_list = []
    
    for n_neighbor in tqdm(n_neighbors, position=0, leave=True, disable=hide_bar):
        
        if dis_metric=='euclidian':
            adj_matrix = kneighbors_graph(data_to_graph, n_neighbors=n_neighbor, n_jobs=-1)
        elif dis_metric=='mahalanobis':
            adj_matrix = kneighbors_graph(data_to_graph, n_neighbors=n_neighbor, n_jobs=-1, metric=dis_metric,
                                          metric_params={'VI': np.linalg.inv(np.cov(data_to_graph.T))})
        else:
            adj_matrix = kneighbors_graph(data_to_graph, n_neighbors=n_neighbor, n_jobs=-1, metric=dis_metric)
          
        adj_matrix = ((adj_matrix+adj_matrix.T)>0).astype(int)
        adj_matrix_csr = adj_matrix.tocsr()
        
        ptr, neighs = adj_matrix_csr.indptr, adj_matrix_csr.indices

        all_walks = biased_random_walk(ptr, neighs, num_steps=random_walk_steps, 
                                       num_walks=num_walks, p=p, q=q, 
                                       nthreads=nthreads, seed=np.random.randint(1,1e6))

        S = estimate_entropy_from_walks(time_series, all_walks, d, tau, complexity)
        
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
