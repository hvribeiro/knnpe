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

knnp: A Python package for characterizing unstructured data with the nearest neighbor permutation entropy
=========================================================================================================

``knnpe``: A Python package for characterizing unstructured data with the nearest neighbor permutation entropy[#voltarelli2024]_


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

- $k$-nearest neighbor permutation entropy[#voltarelli2024]_

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

References
==========

.. [#voltarelli2024] L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, 
   R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro. Characterizing unstructured 
   data with the nearest neighbor permutation entropy. ?? 31, 063110 (2024).