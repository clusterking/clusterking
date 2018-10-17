# Clustering of B to D tau nu kinematical shapes


Requires flavio https://flav-io.github.io



### Output (global_results.out): 

 
* dGq2normtot(epsL,0,0, epsSL, epsT,q2): q2 distribution normalized by total,  integral is one by definition 


* I set epsR an epsSR to zero  as the observables are only sensitive to linear combinations  L + R

* columns: epsL, epsSL, epsT, q2, dGq2normtot(epsL,0,0, epsSL, epsT,q2)




### Clustering: 

data_w_label: clustering of global_results.out using as distance Chi^2(p1,p2) =  Sum_q^2 (Gamma_1(p1) - Gamma_2(p2))^2 with flat errors in q^2.  Clustering is performed with SciPy using Hierarchical clustering, choosing 3 clusters.
