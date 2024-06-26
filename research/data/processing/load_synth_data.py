"""
Module for loading yield surface data from raw VPSC data.
For getting formatted data for bingo, see format_for_bingo.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_yield_surfaces_df(n, folder_path, plot=False):
    """
    Pretty much the same code that you gave me in the notebooks. Reads the
    stress-strain and physics out files to get the yield surface data. Assumes
    that the plastic strain measure is constant throughout each yield surface.
    Returns a dataframe of point trajectories (i.e., how each yield point
    evolves as plastic strain does) in deviatoric space.

    :param n: The run number
    :param folder_path: Folder containing run information
    :param plot: Whether to plot the extract yield surfaces or not
    :return: Dataframe containing information about how yield points evolve
    as plastic strain changes
    """
    pcys_file = open(f"{folder_path}/{n}/PCYS.OUT", "r")
    sections = []
    lines_in_file = None
    for i, line in enumerate(pcys_file):
        if line.strip().startswith("S1"):
            sections.append(i)
        lines_in_file = i + 1
    pcys_file.close()

    # get stress and strain info from STR_STR.OUT
    ss_file = open(f"{folder_path}/{n}/STR_STR.OUT", "r")
    strain_sections = []
    stress_sections = []
    for i, line in enumerate(ss_file):
        if line.strip().startswith("Evm"):
            strain_sections.append([])
            stress_sections.append([])
        else:
            line = line.strip().split()
            line = [float(entry) for entry in line]
            strain_sections[-1].append(line[4])  # E33
            stress_sections[-1].append(line[10])  # S_DEV_33
    ss_file.close()
    print(strain_sections)

    yield_surface_point_sets = []
    yield_surface_eps = []
    for i, sec in enumerate(sections):
        rows_to_read = range(sec, lines_in_file) if i == (len(sections) - 1) else range(sec, sections[i+1])

        vpsc_data = pd.read_csv(f"{folder_path}/{n}/PCYS.OUT", sep='\s+', skiprows=lambda x: x not in rows_to_read)

        two_d_dev_points = vpsc_data[["S1", "S2"]].to_numpy()

        if i == 0:
            yield_surface_eps.append(strain_sections[i][0])
            s_dev_33 = stress_sections[i][0]
        else:
            print(i)
            yield_surface_eps.append(strain_sections[i][-1])
            s_dev_33 = stress_sections[i][0]

        # use the same s_dev_33 value for all points in this yield surface
        all_s_dev_33s = np.repeat(np.array([[s_dev_33]]), len(two_d_dev_points), axis=0)

        # combine S1, S2, and S_33 to get deviatoric points
        # might not be sound to do this, but the S_33 component doesn't
        # play a factor in the mapping process anyway
        dev_points = np.hstack((two_d_dev_points, all_s_dev_33s))
        yield_surface_point_sets.append(dev_points)

    yield_surface_rows = []
    for i, yield_surface_point_set in enumerate(yield_surface_point_sets):
        # use the same eps/plastic strain value for all points in this yield surface
        extended_eps = np.repeat(np.array([[yield_surface_eps[i]]]), len(yield_surface_point_set), axis=0)
        point_indices = np.arange(len(yield_surface_point_set)).reshape((-1, 1))
        df_row = np.hstack((point_indices, yield_surface_point_set, extended_eps))
        yield_surface_rows.append(df_row)

    yield_surface_rows = np.array(yield_surface_rows)

    if plot:
        for i, yield_surface_info in enumerate(yield_surface_rows):
            plt.plot(*yield_surface_info[:, 1:3].T, ".-", label=f"yield surface {i}")
        plt.axis("equal")
        plt.xlabel("$S_1$")
        plt.ylabel("$S_2$")
        plt.legend(loc="upper right")
        plt.show()

    trajectories = yield_surface_rows.transpose((1, 0, 2)).reshape((-1, yield_surface_rows.shape[2]))

    yield_surface_df = pd.DataFrame(data=trajectories, columns=["point_index", "x", "y", "z", "eps"])

    return yield_surface_df
