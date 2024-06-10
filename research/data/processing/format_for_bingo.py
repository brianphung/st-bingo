import numpy as np

from research.data.processing.load_vpsc_data import get_yield_surfaces_df
from research.utility.rotations import align_axes_with_pi_plane_rot


def get_bingo_formatted_data(n, folder_path):
    """
    Format run data for bingo by adding nan rows in-between point trajectories
    returned by get_yield_surfaces_df
    :param n: The run number
    :param folder_path: Folder containing run information
    :return: Numpy array of yield surface data formatted for bingo's implicit
    regression function
    """
    df = get_yield_surfaces_df(n, folder_path)

    # rotate from pi-plane projection to principal stress space
    pi_pos_vec = df[["x", "y", "z"]].to_numpy()
    principal_pos_vec = pi_pos_vec @ align_axes_with_pi_plane_rot()
    df["x"] = principal_pos_vec[:, 0].tolist()
    df["y"] = principal_pos_vec[:, 1].tolist()
    df["z"] = principal_pos_vec[:, 2].tolist()

    max_point_index = int(df["point_index"].max())

    # separate df rows out by point trajectory
    point_trajectories = []
    for i in range(max_point_index + 1):
        point_trajectories.append(df[df["point_index"] == i].to_numpy())

    # point trajectories is an array with dimensions
    #   (num points per yield surface x num yield surfaces x 4)
    #   (4 for x, y, z, and eps)
    point_trajectories = np.array(point_trajectories)[:, :, 1:]

    # add nan rows in-between point trajectories
    implicit_formatted_data = []
    for point_trajectory in point_trajectories:
        implicit_formatted_data.extend(point_trajectory.tolist())
        implicit_formatted_data.append(np.full(point_trajectories.shape[2], np.nan).tolist())

    # clip last nan row
    implicit_formatted_data = implicit_formatted_data[:-1]
    implicit_formatted_data = np.array(implicit_formatted_data)

    return implicit_formatted_data


def get_bingo_formatted_data_transposed(n, folder_path):
    """
    Similar to get_bingo_formatted_data but formulates points trajectories
    within yield surfaces rather than in-between them. Bingo needs local
    derivative information about the yield points, this function returns
    the points such that the local derivative is within the yield surface
    rather than between them.
    :param n: The run number
    :param folder_path: Folder containing run information
    :return: Numpy array of yield surface data formatted for bingo's implicit
    regression function, but with 'transposed' point trajectories
    """
    df = get_yield_surfaces_df(n, folder_path)

    # rotate from pi-plane projection to principal stress space
    pi_pos_vec = df[["x", "y", "z"]].to_numpy()
    principal_pos_vec = pi_pos_vec @ align_axes_with_pi_plane_rot()
    df["x"] = principal_pos_vec[:, 0].tolist()
    df["y"] = principal_pos_vec[:, 1].tolist()
    df["z"] = principal_pos_vec[:, 2].tolist()

    max_point_index = int(df["point_index"].max())

    # separate df rows out by point
    point_trajectories = []
    for i in range(max_point_index + 1):
        point_trajectories.append(df[df["point_index"] == i].to_numpy())

    # point trajectories is an array with dimensions
    #   (num points per yield surface x num yield surfaces x 4)
    #   (4 for x, y, z, and eps)
    point_trajectories = np.array(point_trajectories)[:, :, 1:]

    # yield surfaces is an array with dimensions
    #   (num yield surfaces x points per yield surface x 4)
    yield_surfaces = point_trajectories.transpose((1, 0, 2))

    # add nan rows in-between yield surfaces
    implicit_formatted_data = []
    for yield_surface_points in yield_surfaces:
        implicit_formatted_data.extend(yield_surface_points.tolist())
        implicit_formatted_data.append(np.full(yield_surfaces.shape[2], np.nan).tolist())

    # clip last nan row
    implicit_formatted_data = implicit_formatted_data[:-1]
    implicit_formatted_data = np.array(implicit_formatted_data)

    return implicit_formatted_data


if __name__ == '__main__':
    runs = [75]
    raw_data_path = f"../raw_data/YS_evo"

    for n in runs:
        normal_data = get_bingo_formatted_data(n, raw_data_path)
        transposed_data = get_bingo_formatted_data_transposed(n, raw_data_path)

        np.savetxt(f"../processed_data/vpsc_{n}_bingo_format.txt", normal_data)
        np.savetxt(f"../processed_data/vpsc_{n}_transpose_bingo_format.txt", transposed_data)
