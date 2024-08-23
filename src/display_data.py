#%%
from file_operator.yaml_operation import read_yaml
import numpy as np
# from typing import NDArray
# import ty
from phase_field_2d_ternary.matrix_plot_tools import Ternary
import matplotlib.pyplot as plt
from phase_field_2d_ternary.phase_field import PhaseField2d3c

# datas = read_yaml("summary/used_in_NN.yaml")
datas = read_yaml("summary/result_summary.yaml")
# x = list(map(lambda x: np.transpose(np.load(x["persistent_img_path"]), (1,2,0)), datas))


for i, data in enumerate(datas):
    if i > 100:
        break
    con1 = np.load(data["file_list"][-1][0])
    con2 = np.load(data["file_list"][-1][1])
    print(data)
    w12 = data["information"]["w12"]
    w23 = data["information"]["w23"]
    w13 = data["information"]["w13"]

    a = PhaseField2d3c(w12, w13, w23)
    a.plot_ternary_contour_and_composition(con1, con2)

    plt.show()
    Ternary.imshow3(con1, con2)
    plt.show()

    # d = np.load(data["persistent_img_path_hom1"])
    # plot_persistence_image_from_tda_diagram(d[0])

    plt.show()
#%%

