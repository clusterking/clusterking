Clustering of B to D tau nu kinematical shapes
==============================================

|Build Status| |Doc Status|

.. |Build Status| image:: https://travis-ci.org/RD-clustering/B_decays_clustering.svg?branch=master
   :target: https://travis-ci.org/RD-clustering/B_decays_clustering

.. |Doc Status| image:: https://readthedocs.org/projects/bclustering/badge/?version=latest
   :target: https://bclustering.readthedocs.io/en/latest/
   :alt: Documentation Status

.. start-body

Installation
------------

.. code:: sh

    pip3 install --user git+https://github.com/wilson-eft/wilson.git
    git clone git@github.com:RD-clustering/B_decays_clustering.git
    cd B_decays_clustering
    pip3 install --user .

Usage
-----

Good starting point: Jupyter notebooks in the ``jupyter`` directory.

Step 1: Build histograms
~~~~~~~~~~~~~~~~~~~~~~~~

Build q2 histograms for the different NP benchmark points.
This can be done with the command line interface:

.. code:: sh

    bclustering/bin/scan --np-grid-subdivision 3 --grid-subdivision 5 --output output/scan/quick

More information on the command line options can be found by running
``scan.py --help``.

This produces two output files:

- ``output/scan/quick_data.csv`` holds the data as a CSV file with the 
  columns ``index`` (number of the benchmark point), 
  ``l``, ``r``, ``sr``, ``sl``, ``t`` (the five Wilson coefficients),
  ``bin0``, ``bin1``, ..., ``bin4`` (the five bins of the q2 
  distribution). 
    
- ``output/scan/quick_metadata.json`` holds metadata (e.g. how many
  bins have we used, which software version etc.).
  It's a prettified json file, so it's pretty human readable.


Step 2: Clustering
~~~~~~~~~~~~~~~~~~
    
This can be done with the command line interface as well: 

.. code::bash

    bclustering/bin/cluster --input output/scan/quick.out --output output/cluster/quick

This again produces two output files:

- ``output/scan/quick_data.csv`` containing the same columns as 
  ``output/scan/quick_data.csv`` plus an additional column ``cluster``
  that contains the number of the cluster
    
- ``output/scan/quick_metadata.json`` combined metadata of step 1 and
  this step.
    
Furthermore, a dendrogram is produced automatically and saved at
``output/cluster/quick_dend.pdf``. Our example: 

.. figure:: https://raw.githubusercontent.com/celis/B_decays_clustering/master/readme_assets/quick_dend.png?raw=true)
    :alt: Dendrogram
    :align: center

.. end-body
