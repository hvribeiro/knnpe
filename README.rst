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

- $k$-nearest neighbor permutation entropy [#voltarelli2024]_

For more detailed information about the methods implemented in ``knnpe``, please 
consult its `documentation <https://hvribeiro.github.io/knnpe/_build/html/index.html>`_.

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

Whether the dependencies are available or nor, ``knnpe`` can be installed via:

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

References
==========

.. [#voltarelli2024] L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, 
   R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro. Characterizing unstructured 
   data with the nearest neighbor permutation entropy. ?? 31, 063110 (2024).
.. [#bandtpompe2002] C. Bandt, B. Pompe. Permutation entropy: A Natural 
   Complexity Measure for Time Series. Physical Review Letters 88, 174102 (2002).
.. [#ribeiro2012] H. V. Ribeiro, L. Zunino, E. K. Lenzi, P. A. Santoro, R. S.
   Mendes. Complexity-Entropy Causality Plane as a Complexity
   Measure for Two-Dimensional Patterns. PLOS ONE 7, e40689 (2012).
