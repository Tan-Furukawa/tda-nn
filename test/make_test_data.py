
#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#%%
from file_operator.yaml_operation import yaml_dump, read_yaml
from file_operator.base import save_str, create_directory
from src.file_operatoration import extract_directory_information
from tda_for_phase_field.random_sampling import (
    random_sampling_from_matrices,
    npMap,
    select_specific_phase,
)

create_directory("tmp")
li = extract_directory_information("ph_result")
save_str(f"tmp/phase_field_result.yaml", yaml_dump(li))

#%%
from src.classify_phase import classify_as_six_phase, make_sample_of_classify_as_six_phase, classify_as_three_phase
import numpy as np
from phase_field_2d_ternary.matrix_plot_tools import Ternary
from phase_field_2d_ternary import PhaseField2d3c
import matplotlib.pyplot as plt

datas = read_yaml("tmp/phase_field_result.yaml")

for i, data in enumerate(datas):
    if i > 100:
        break
    con1 = np.load(data["file_list"][-1][0])
    con2 = np.load(data["file_list"][-1][1])
    w12 = data["information"]["w12"]
    w23 = data["information"]["w23"]
    w13 = data["information"]["w13"]
    rcon = random_sampling_from_matrices([con1, con2], 10000)

    rx = npMap(lambda x: x[0], rcon)
    ry = npMap(lambda x: x[1], rcon)
    rcon1 = npMap(lambda x: x[2], rcon)
    rcon2 = npMap(lambda x: x[3], rcon)
    phase = classify_as_three_phase(rcon1, rcon2)

    a = PhaseField2d3c(w12, w13, w23)
    a.plot_ternary_contour_and_composition(con1, con2)
    plt.show()

    # Ternary.imshow3(con1, con2)
    plt.scatter(ry, rx, c=phase ,s = 1)
    plt.show()

make_sample_of_classify_as_six_phase()
#%%