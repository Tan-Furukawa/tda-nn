## %%
import numpy as np
from phase_field_2d import PhaseField
import random
from numpy.typing import NDArray
from phase_field_2d_ternary.matrix_plot_tools import Ternary
import matplotlib.pyplot as plt
from phase_field_2d_ternary import PhaseField2d3c
from combine import random_make_parameter

# matrix = np.identity(3)
# matrix[2, 2] = 1

parameter_generator = random_make_parameter(
    {
        "eta22": (1, 3.5),
        "eta12": (0, 1.4),
        "w": (2.1, 5),
        "c0": (0.25, 0.75),
    }
)
# %%

for i, p in enumerate(parameter_generator):
    print(p)
    eta = np.array([1, (p["eta22"]-1)**2+1, 0])
    try:
        phase_field = PhaseField(
            p["w"],
            p["c0"],
            method="anisotropic",
            eta=eta,
            save_dir_name="summary/raw_data/result_2d_fix",
            record=True,
        )
        phase_field.nprint = 10000
        phase_field.nsave = 2000
        phase_field.nstep = 10000
        phase_field.dtime = 0.005
        phase_field.seed = i
        phase_field.start()
    except:
        continue
# %%


# %%
