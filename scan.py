#!/usr/bin/env python3



from distribution import *

import time
import datetime


###
### scans the NP parameter space in a grid and also q2, producing the normalized q2 distribution
###   I guess this table can then be used for the clustering algorithm, any.  



# Output file
output = "global_results.out"

# Number of q2 grid steps
grid_subdivisions = 15

# Number of NP grid steps
np_grid_subdivisions = 20




# Prepare output
fresults = open(output, 'w')



## q2 distribution normalized by total,  integral of this would be 1 by definition

## dGq2normtot(epsL, epsR, epsSR, epsSL, epsT,q2):


##  I set epsR an epsSR to zero  as the observables are only sensitive to linear combinations  L + R

performed_integrations = 0
total_integrations = np_grid_subdivisions ** 4

for epsL in np.linspace(-0.30, 0.30, np_grid_subdivisions):
    for epsSL in np.linspace(-0.30, 0.30, np_grid_subdivisions):
        for epsT in np.linspace(-0.40, 0.40, np_grid_subdivisions):
            for q2 in np.linspace(q2min,q2max, grid_subdivisions):
                start = time.time()
                dist_tmp = dGq2normtot(epsL, 0, 0, epsSL, epsT,q2)
                elapsed_time = time.time() - start
                remaining_time = (total_integrations - performed_integrations) * elapsed_time
                fresults.write('%.5f    '%epsL  + '%.5f    '%epsSL  + '%.5f  '%epsT  + '%.5f   '%q2  + '%.10f       '%dist_tmp + '\n')
                performed_integrations += 1
                print("Progress: {}/{} ({:.2f}%). Last timing: {:.1f}s => "
                      "time remaining: {}".format(
                        performed_integrations,
                        total_integrations,
                        100*performed_integrations/total_integrations,
                        elapsed_time,
                        datetime.timedelta(seconds=remaining_time)
                ))


fresults.close()

print("***** scan finalized *****")

