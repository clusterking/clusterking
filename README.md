# Clustering of B to D tau nu kinematical shapes

## Installation

* Requires flavio https://flav-io.github.io

## Usage

*   Step 1: Build q2 histograms for the different NP benchmark points.
    Quick example:
        
        ./scan.py -n 3 -g 5 -o output/scan/quick.out

    Calculate q2 histograms with 5 bins in q2, sampling the NP parameters
    ``epsL``, ``epsSL`` and ``epsT`` with 3 points. 
    
    More information on the command line options for ``scan.py``:
    ``scan.py --help``.
    
    The output file is currently in csv format.
    
*   Step 2: Build distance matrix from the q2 histograms.
    Quick example using our output from step 1:
    
        ./distance_matrix.py -i output/scan/quick.out -o output/distance/quick.out
      
    More information on the command line options for ``distance_matrix.py``:
    ``distance_matrix.py --help``. 
    
*   Step 3: Build clusters: Not implemented yet?

        ./cluster.py -i output/distance/quick.out -o output/cluster/quick.out
       
## Clustering Result: 

data_w_label: clustering using as distance Chi^2(p1,p2) =  Sum_q^2 (Gamma_1(p1) - Gamma_2(p2))^2 with flat errors in q^2.  Clustering is performed with SciPy using Hierarchical clustering, choosing 3 clusters.
