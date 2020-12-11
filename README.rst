.. note: Always use full path to image, from https://raw.githubusercontent.com/
   because it won't render on pypi and others otherwise if you use the relative
   path from this repo :(

.. image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/logo/logo.png
   :align: right

Clustering of Kinematic Graphs
==============================

|Build Status| |Coveralls| |Doc Status| |Pypi status| |Binder| |Chat| |License| |Black|

.. |Build Status| image::  https://github.com/clusterking/clusterking/workflows/testing/badge.svg
   :target: https://github.com/clusterking/clusterking/actions
   :alt: CI

.. |Coveralls| image:: https://coveralls.io/repos/github/clusterking/clusterking/badge.svg?branch=master
   :target: https://coveralls.io/github/clusterking/clusterking?branch=master

.. |Doc Status| image:: https://readthedocs.org/projects/clusterking/badge/?version=latest
   :target: https://clusterking.readthedocs.io/
   :alt: Documentation

.. |Pypi Status| image:: https://badge.fury.io/py/clusterking.svg
   :target: https://pypi.org/project/clusterking/
   :alt: Pypi

.. |Binder| image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/badges/png/binder.png
   :target: https://mybinder.org/v2/gh/clusterking/clusterking/master?filepath=examples%2Fjupyter_notebooks
   :alt: Binder

.. |Chat| image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/badges/png/gitter.png
   :target: https://gitter.im/clusterking/community
   :alt: Gitter

.. |License| image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/badges/png/license.png
   :target: https://github.com/clusterking/clusterking/blob/master/LICENSE.txt
   :alt: License

.. |Black| image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/badges/png/black.png
   :target: https://github.com/python/black
   :alt: Black

.. start-body

Description
-----------

This package provides a flexible yet easy to use framework to cluster sets of
histograms (or other higher dimensional data) and to select benchmark points
representing each cluster. The package particularly focuses on use cases in
high energy physics.

A physics use case has been demonstrated in https://arxiv.org/abs/1909.11088.

Physics Case
------------

While most of this package is very general and can be applied to a broad variety
of use cases, we have been focusing on applications in high energy physics
(particle physics) so far and provide additional convenience methods for this
use case. In particular, most of the current tutorials are in this context.

Though very successful, the Standard Model of Particle Physics is believed to
be uncomplete, prompting the search for New Physics (NP). The phenomenology
of NP models typically depends on a number of free parameters, sometimes
strongly influencing the shape of distributions of kinematic variables.
Besides being an obvious challenge when presenting exclusion limits on such
models, this also is an issue for experimental analyses that need to make
assumptions on kinematic distributions in order to extract features of
interest, but still want to publish their results in a very general way.

By clustering the NP parameter space based on a metric that quantifies the
similarity of the resulting kinematic distributions, a small number of NP
benchmark points can be chosen in such a way that they can together represent
the whole parameter space. Experiments (and theorists) can then report
exclusion limits and measurements for these benchmark points without
sacrificing generality.

Installation
------------

``clusterking`` can be installed/upgraded with the `python package installer <https://pip.pypa.io/en/stable/>`_:

.. code:: sh

    pip3 install --user --upgrade "clusterking[plotting]"

If you do not require plotting, you can remove ``[plotting]``.

More options and troubleshooting advice is given in the `documentation <https://clusterking.readthedocs.io/en/latest/installation.html>`_.

Caveats
-------

* Version 1.0.0 contained several mistakes in the chi2 metric. Please make sure
  that you are at least using versoin 1.1.0. These mistakes were also found in
  the `paper <https://arxiv.org/abs/1909.11088>`_ and will be fixed soon.

Usage and Documentation
-----------------------

Good starting point: **Jupyter notebooks** in the ``examples/jupyter_notebook`` directory.
You can also try running them online right now (without any installation required) using
|binder2|_ (just note that this is somewhat unstable, slow and takes some time
to start up).

.. |binder2| replace:: binder
.. _binder2: https://mybinder.org/v2/gh/clusterking/clusterking/master?filepath=examples%2Fjupyter_notebooks

.. _run online using binder: https://mybinder.org/v2/gh/clusterking/clusterking/master?filepath=examples%2Fjupyter_notebooks

For a documentation of the classes and functions in this package, **read the docs on** |readthedocs.io|_.

.. |readthedocs.io| replace:: **readthedocs.io**
.. _readthedocs.io: https://clusterking.readthedocs.io/

For additional examples, presentations and more, you can also head to our `other repositories`_.

.. _other repositories: https://github.com/clusterking

Example
-------

Sample
~~~~~~

The following code (taken from ``examples/jupyter_notebook/010_basic_tutorial.ipynb``) is all that
is needed to cluster the shape of the ``q^2`` distribution of ``B -> D tau nu``
in the space of Wilson coefficients:

.. code:: python

   import flavio
   import numpy as np
   import clusterking as ck

   s = ck.scan.WilsonScanner(scale=5, eft='WET', basis='flavio')

   # Set up kinematic function

   def dBrdq2(w, q):
       return flavio.np_prediction("dBR/dq2(B+->Dtaunu)", w, q)

   s.set_dfunction(
       dBrdq2,
       binning=np.linspace(3.2, 11.6, 10),
       normalize=True
   )

   # Set sampling points in Wilson space

   s.set_spoints_equidist({
       "CVL_bctaunutau": (-1, 1, 10),
       "CSL_bctaunutau": (-1, 1, 10),
       "CT_bctaunutau": (-1, 1, 10)
   })

   # Create data object to write to and run

   d = ck.DataWithErrors()
   r = s.run(d)
   r.write()  # Write results back to data object

Cluster
~~~~~~~

Using hierarchical clustering:

.. code:: python

   c = ck.cluster.HierarchyCluster()  # Initialize worker class
   c.set_metric("euclidean")
   c.set_max_d(0.15)      # "Cut off" value for hierarchy
   r = c.run(d)           # Run clustering on d
   r.write()              # Write results to d

Benchmark points
~~~~~~~~~~~~~~~~

.. code:: python

   b = ck.Benchmark() # Initialize worker class
   b.set_metric("euclidean")
   r = b.run(d)        # Select benchmark points based on metric
   r.write()           # Write results back to d

Plotting
~~~~~~~~

.. code:: python

    d.plot_clusters_scatter(
        ['CVL_bctaunutau', 'CSL_bctaunutau', 'CT_bctaunutau'],
        clusters=[1,2]  # Only plot 2 clusters for better visibility
    )

.. image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/plots/scatter_3d_02.png

.. code:: python

    d.plot_clusters_fill(['CVL_bctaunutau', 'CSL_bctaunutau'])

.. image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/plots/fill_2d.png

Plotting all benchmark points:

.. code:: python

    d.plot_dist()

.. image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/plots/all_bcurves.png

Plotting minima and maxima of bin contents for all histograms in a cluster (+benchmark histogram):

.. code:: python

    d.plot_dist_minmax(clusters=[0, 2])

.. image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/plots/minmax_02.png

Similarly with box plots:

.. code:: python

   d.plot_dist_box()

.. image:: https://raw.githubusercontent.com/clusterking/clusterking/master/readme_assets/plots/box_plot.png

License & Contributing
----------------------

This project is ongoing work and questions_, comments,
`bug reports`_ or `pull requests`_ are most welcome. You can also use the chat
room on gitter_ or contact us via email_.
We are also working on a paper, so please make sure to cite us once we publish.

.. _email: mailto:clusterkinematics@gmail.com
.. _gitter: https://gitter.im/clusterking/community
.. _questions: https://github.com/clusterking/clusterking/issues
.. _bug reports: https://github.com/clusterking/clusterking/issues
.. _pull requests: https://github.com/clusterking/clusterking/pulls

This software is licenced under the `MIT license`_.

.. _MIT  license: https://github.com/clusterking/clusterking/blob/master/LICENSE.txt

.. end-body
