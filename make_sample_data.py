# %%
import numpy as np
from ph_used import PhaseFieldBreakByInterfaceEnergy
import random
from numpy.typing import NDArray
from tda_for_phase_field.random_samplig import (
    random_sampling_from_matrices,
    npMap,
    select_specific_phase,
)
import tda_for_phase_field.tda as tda
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
        "c20": (0.2, 0.4),
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
        )
        phase_field.dtime = 0.003
        phase_field.nprint = 100000
        phase_field.nsave = 15000
        phase_field.nstep = 60001
        phase_field.start()
    except:
        continue
# %%


con1 = np.load("result/output_2024-08-05-12-19-32/con1_60.npy")
con2 = np.load("result/output_2024-08-05-12-19-32/con2_60.npy")

res = random_sampling_from_matrices([con1, con2], 1000)
res = select_specific_phase(res, 1)
coordinate_of_points = npMap(lambda x: [x[0], x[1]], res)
x = npMap(lambda x: x[0], res)
y = npMap(lambda x: x[1], res)

tda_res = tda.make_tda_diagram(coordinate_of_points, dim0_hole=False)
# tda_res = npFilter(lambda x: x[0] > 5, tda_res)
img = tda.get_persistent_image_info(tda_res)
tda.plot_persistence_image(img)
plt.show()
tda.plot_persistent_diagram(coordinate_of_points)
plt.show()

Ternary.imshow3(con1, con2)
plt.show()
plt.scatter(y, x, s=2)
plt.show()

# %%
