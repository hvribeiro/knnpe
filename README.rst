.. |logo1| image:: https://img.shields.io/pypi/v/knnpe?style=plastic   :alt: PyPI 
   :target: https://pypi.org/project/knnpe/
   :scale: 100%
.. |logo2| image:: https://img.shields.io/github/license/hvribeiro/knnpe?style=plastic   :alt: GitHub 
   :target: https://github.com/hvribeiro/knnpe/blob/master/LICENSE
   :scale: 100%
.. |logo3| image:: https://img.shields.io/pypi/dm/knnpe?style=plastic   :alt: PyPI - Downloads
   :target: https://pypi.org/project/knnpe/
   :scale: 100%
.. |logo4| image:: https://readthedocs.org/projects/knnpe/badge/?version=latest
   :target: https://knnpe.readthedocs.io/?badge=latest
   :alt: Documentation Status
   :scale: 100%

|logo1| |logo2| |logo3| |logo4|

knnpe: A Python package implementing the *k*-nearest neighbor permutation entropy
=================================================================================

The *k*-nearest neighbor permutation entropy [#voltarelli2024]_ extends the fundamental premise of investigating 
the relative ordering of time series elements [#bandtpompe2002]_ or image pixels [#ribeiro2012]_ inaugurated by 
permutation entropy to unstructured datasets. This method builds upon nearest neighbor graphs to establish neighborhood
relations among data points and uses random walks over these graphs to extract ordinal patterns and their distribution, 
thereby defining the $k$-nearest neighbor permutation entropy.

If you have used ``knnpe`` in a scientific publication, we would appreciate citations to the following reference [#voltarelli2024]_:

- L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro, 
  `Characterizing unstructured data with the nearest neighbor permutation entropy <https://doi.org/?>`_, 
  ?, ? (2024).  `arXiv:? <https://arxiv.org/abs/?>`_

.. code-block:: bibtex
    
   @article{voltarelli2024characterizing,
    title         = {Characterizing unstructured data with the nearest neighbor permutation entropy}, 
    author        = {L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro},
    journal       = {},
    volume        = {},
    number        = {},
    pages         = {},
    year          = {2024},
    doi           = {},
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

If these dependencies are not available, ``knnpe`` will use a native Python function for doing the random walks. This function is also parallelized and may work nicely for most applications; still, it is significantly slower than its C counterpart. For large datasets, we strongly recommend using the C version.

Whether the dependencies are available or not, ``knnpe`` can be installed via:

.. code-block:: console

   pip install knnpe

or you can directly clone its git repository:

.. code-block:: console

   git clone https://github.com/hvribeiro/knnpe.git
   cd knnpe
   pip install -e .


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

The last column in `data` corresponds to $\\{z_i\\}_{i=1,\\dots,N}$ values, while the first two columns are used as the data coordinates $\\vec{r}_i = (x_i,y_i)$. If the dataset has more dimensions in data coordinates, they must be passed as the first columns of the dataset, and the last column is always assumed to be corresponding $z_i$ values. The code below illustrates the case of data with three dimensions in data coordinates:

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

We provide a `notebook <https://github.com/hvribeiro/knnpe/blob/master/examples/knnpe.ipynb>`_
illustrating how to use ``knnpe`` and further information we refer to the knnpe's `documentation <https://readthedocs.org/projects/knnpe/badge/?version=latest>`_

Contributing
============

Pull requests addressing errors or adding new functionalities are always welcome.

References
==========

.. [#voltarelli2024] L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, 
   R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro. Characterizing unstructured 
   data with the nearest neighbor permutation entropy. ?, ? (2024).
.. [#bandtpompe2002] C. Bandt, B. Pompe. Permutation entropy: A Natural 
   Complexity Measure for Time Series. Physical Review Letters 88, 174102 (2002).
.. [#ribeiro2012] H. V. Ribeiro, L. Zunino, E. K. Lenzi, P. A. Santoro, R. S.
   Mendes. Complexity-Entropy Causality Plane as a Complexity
   Measure for Two-Dimensional Patterns. PLOS ONE 7, e40689 (2012).
