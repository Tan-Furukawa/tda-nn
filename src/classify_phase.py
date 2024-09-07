#%%
import numpy as np
from phase_field_2d_ternary.matrix_plot_tools import Ternary
import matplotlib.pyplot as plt
import numpy as np
from phase_field_2d_ternary.matrix_plot_tools import Ternary
import matplotlib.pyplot as plt

def determine_three_phase(con1, con2):
    con1_new = np.zeros_like(con1)
    con2_new = np.zeros_like(con2)
    con3 = -con1 - con2 + 1
    con1_new[np.logical_and(con1 >= con3, con1 >= con2)] = 1
    con2_new[np.logical_and(con2 >= con3, con2 > con1)] = 1
    return con1_new, con2_new

def classify_as_three_phase(c1, c2):
    c3 = -c1 - c2 + 1
    phase = np.zeros_like(c1)
    phase[(c2 <= c1) & (c3 <= c1)] = 1
    phase[(c1 <= c2) & (c3 <= c2)] = 2
    phase[(c1 <= c3) & (c2 <= c3)] = 3
    return phase

def classify_as_six_phase(c1, c2, threshold = 1/4):
    c3 = -c1 - c2 + 1
    d = 0.00001 # avoid 0 division
    p1 = c1 / (c1 + c2 + d)
    p2 = c2 / (c2 + c3 + d)
    p3 = c3 / (c3 + c1 + d)
    phase = np.zeros_like(c1)
    phase[ (c1 >= c3) & (c2 >= c3) ] = -3
    phase[
        (c3 >= c1) & (c2 >= c1)
    ] = -1
    phase[
        (c3 >= c2) & (c1 >= c2)
    ] = -2
    phase[
        ((c3 >= c2) & (c1 >= c2) & (p3 <= threshold))|
        ((c2 >= c3) & (c1 >= c3) & (p1 >= 1-threshold))
    ] = 1
    phase[
        ((c3 >= c1) & (c2 >= c1) & (p2 >= 1-threshold)) |
        ((c2 >= c3) & (c1 >= c3) & (p1 <= threshold))
    ] = 2
    phase[
        ((c3 >= c1) & (c2 >= c1) & (p2 <= threshold)) |
        ((c3 >= c2) & (c1 >= c2) & (p3 >= 1-threshold))
    ] = 3
    return phase

def make_sample_of_classify_as_three_phase():
    Ternary.plot_blank_ternary()
    c1, c2 = Ternary.generate_triangle_mesh(100)
    x, y = Ternary.convert_ternary_to_Cartesian_coordinate(c1, c2)
    phase = classify_as_three_phase(c1,c2)
    colors = plt.colormaps['Set2']
    scatter = plt.scatter(x, y, c=phase, cmap=colors, s=1)
    legend1 = plt.legend(*scatter.legend_elements(), title="c")
    plt.gca().add_artist(legend1)

def make_sample_of_classify_as_six_phase(threshold = 1/4):
    Ternary.plot_blank_ternary()
    c1, c2 = Ternary.generate_triangle_mesh(100)
    x, y = Ternary.convert_ternary_to_Cartesian_coordinate(c1, c2)
    phase = classify_as_six_phase(c1,c2,threshold)
    colors = plt.colormaps['Set2']
    scatter = plt.scatter(x, y, c=phase, cmap=colors, s=1)
    legend1 = plt.legend(*scatter.legend_elements(), title="c")
    plt.gca().add_artist(legend1)

if __name__=="__main__":
    make_sample_of_classify_as_six_phase()
    plt.show()
    make_sample_of_classify_as_three_phase()
    plt.show()