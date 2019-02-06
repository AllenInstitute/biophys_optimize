import sys, os

import allensdk.core.json_utilities as ju

storage_dir = "./test_storage"
fit_types = sys.argv[1:-1]
seed = 1234

data = {
    'paths': {
        'fits': [ {
                "fit_type": ft,
                "hof_fit": os.path.join(storage_dir, "%s_%d_final_hof_fit.txt" % (ft, seed)),
                "hof": os.path.join(storage_dir, "%s_%d_final_hof.txt" % (ft, seed))
                } for ft in fit_types ] } }
    

ju.write(sys.argv[-1], data)
