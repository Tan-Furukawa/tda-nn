## %%
import numpy as np
from ph_used import PhaseFieldBreakByInterfaceEnergy
import random
from numpy.typing import NDArray
from phase_field_2d_ternary.matrix_plot_tools import Ternary
import matplotlib.pyplot as plt
from phase_field_2d_ternary import PhaseField2d3c
from combine import random_make_parameter


def convert_ternary_gradient_parameter_to_binary(
    mat: NDArray,
) -> NDArray:
    k11 = 2 * (mat[0][0] - mat[0][2] + mat[2][2])
    k22 = 2 * (mat[1][1] - mat[1][2] + mat[2][2])
    k12 = 2 * mat[2][2] + mat[0][1] - mat[0][2] - mat[1][2]

    return np.array([k11, k22, k12])


matrix = np.identity(3)
matrix[2, 2] = 1
k = convert_ternary_gradient_parameter_to_binary(matrix)

parameter_generator = random_make_parameter(
    {
        "k11": (8,16),
        "k22": (8,16),
        "k12": (2,8),
        "w12": (1.9, 5),
        "w13": (1.9, 5),
        "w23": (1.9, 5),
        "c10": (0.2, 0.4),
        # "c10": (0.3333, 0.3333),
        "c20": (0.2, 0.4),
        # "c20": (0.3333, 0.3333),
        "L12": (-2, 0),
        "L13": (-2, 0),
        "L23": (-2, 0),
    }
)

for p in parameter_generator:
    print(p)
    try:
        phase_field = PhaseFieldBreakByInterfaceEnergy(
            10000,
            k11=p["k11"],
            k22=p["k22"],
            k12=p["k12"],
            w12=p["w12"],
            w13=p["w13"],
            w23=p["w23"],
            c10=p["c10"],
            c20=p["c20"],
            L12=p["L12"],
            L13=p["L13"],
            L23=p["L23"],
            record=True,
            # save_dir_name = "result_const_c0_033333"
            save_dir_name = "summary/raw_data/result_12000"
        )
        phase_field.dtime = 0.003
        phase_field.nprint = 100000
        phase_field.nsave = 15000
        phase_field.nstep = 120001
        phase_field.start()
    except :
        continue
# %%


# %%
