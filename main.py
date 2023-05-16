# file used to run main program

from utils import *

#create 3d  (contains Nan values)
data = load_data("data/2022-12-08-rat_kidney.npy")

# plot_slice(data,1)
# plot_spect(data,([200,200]))
# create 2D data
data2D = load_data("data/2022-12-08-rat_kidney.npy",to3D=False)

import numpy as np
randomspec = np.random.randint(0,len(data2D),size=5)
plot_spect(data2D,randomspec)