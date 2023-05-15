# file used to run main program

from utils import *

#create 3d data
data = load_data("data/2022-12-08-rat_kidney.npy")
# plot_slice(data,1)

# plot_spect(data,([200,200]))

data2D = load_data("data/2022-12-08-rat_kidney.npy",to3D=False)

import numpy as np
print(np.shape(data2D))
plot_spect(data2D,600)