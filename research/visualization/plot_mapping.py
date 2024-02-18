from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from research.utility.rotations import align_pi_plane_with_axes_rot


def hill_mapping_model(eps):
    from numpy import sin, cos, array

    # bingo uses sqrt(|x|) as sqrt(x)
    def sqrt(x):
        return np.sqrt(np.abs(x))

    X_0 = eps
    if not isinstance(eps, np.ndarray):
        eps = np.array([eps])[:, None, None]

    # input your individual here
    eq = X_0 * array([[4.84938514, -1.39136693, 3.57891267],
                      [0.92418741, 2.16747212, 3.7473522],
                      [2.52797065, 1.72322281, 2.07687239]])

    scaling_factor = 1.0
    # set scaling factor if needed (e.g., if average scale is too large)
    eq *= scaling_factor

    if eq.shape[0] != eps.shape[0]:
        eq = np.repeat(eq[None, :, :], eps.shape[0], axis=0)

    return eq


def vpsc_57_mapping_model(eps):
    from numpy import sin, cos, array

    # bingo uses sqrt(|x|) as sqrt(x)
    def sqrt(x):
        return np.sqrt(np.abs(x))

    X_0 = eps
    if not isinstance(eps, np.ndarray):
        eps = np.array([eps])[:, None, None]

    # input your individual here
    eq = (0.00022751 * X_0 + 5.29761659) * \
         array([[-0.22488406, 0.39036446, 0.23017129],
                [-0.09263297, 0.41016726, 0.07469385],
                [-0.25825709, 0.51208763, 0.13926473]])

    scaling_factor = 1.0
    # set scaling factor if needed (e.g., if average scale is too large)
    eq *= scaling_factor

    if eq.shape[0] != eps.shape[0]:
        eq = np.repeat(eq[None, :, :], eps.shape[0], axis=0)

    return eq


def get_yield(principal_stress, *, eps=0.0, mapping_model):
    """
    Get real yield stress values as a function of plastic strain using the
    mapping provided above
    """
    if len(principal_stress.shape) == 2:
        principal_stress = principal_stress[None, :, :]

    if isinstance(eps, np.ndarray) and eps.ndim == 2:
        eps = np.expand_dims(eps, 0)

    mapping_matrix = mapping_model(eps)

    P_fict = np.array([[1, -0.5, -0.5],
                       [-0.5, 1, -0.5],
                       [-0.5, -0.5, 1]])

    P_real = mapping_matrix.transpose((0, 2, 1)) @ P_fict @ mapping_matrix
    yield_values = principal_stress.transpose((0, 2, 1)) @ P_real @ principal_stress
    return yield_values.squeeze()


def get_points_from_yield(yield_fn, coord_ranges, level_to_plot, coord_n=1000):
    """
    Draw yield contours and get points from those contours.

    :param yield_fn: Function that returns yield stress values as a function
    of principal stresses
    :param coord_ranges: [min, max] range of meshgrid to sample over
    :param level_to_plot: Contour levels to plot
    :param coord_n: Number of to separate
    :return: Yield surface points on the pi plane defined by the plotted
    contour, points, and a matplotlib contour object
    representing the contour, cs
    """
    x = np.linspace(*coord_ranges, num=coord_n)
    y = np.linspace(*coord_ranges, num=coord_n)
    z = np.linspace(*coord_ranges, num=coord_n)

    # get meshgrid
    new_x, new_y = np.meshgrid(x, y)
    _, new_z = np.meshgrid(y, z)

    pos_vec = np.dstack([new_x, new_y, new_z])
    pi_plane_pos = pos_vec @ align_pi_plane_with_axes_rot()

    # evaluate yield surface at meshgrid points
    h = np.zeros((pos_vec.shape[0], pos_vec.shape[1]))
    for i in range(pos_vec.shape[0]):
        for j in range(pos_vec.shape[1]):
            h[i][j] = yield_fn(pi_plane_pos[i][j].reshape((3, 1)))

    cs = plt.contour(new_x, new_y, h, levels=[level_to_plot])

    try:
        # extract points from contours
        points = []
        for path in cs.collections[0].get_paths():
            points.extend(path.vertices)
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        z = level_to_plot

        points = np.vstack([x, y, np.ones_like(x) * z]).T
        return points, cs
    except IndexError:  # if the contour has no points
        return np.array([]), cs


def draw_principal_axes(ax, length_of_axes=2.5, scale=100):
    """
    Draws principal axes on graph, change length of axes and scale
    """
    zero_pos = np.zeros(2)
    straight_up_pos = np.array([0, 1]) * length_of_axes
    bottom_right_pos = np.array([np.cos(-np.pi / 6.), np.sin(-np.pi / 6.)]) * length_of_axes
    bottom_left_pos = np.array([np.cos(7 * np.pi / 6.), np.sin(7 * np.pi / 6.)]) * length_of_axes

    ax.quiver(*zero_pos, *straight_up_pos, scale_units="xy", scale=1)
    ax.annotate(r"$\sigma_2$", straight_up_pos + np.array([0.25, -0.3]) * scale)

    ax.quiver(*zero_pos, *bottom_right_pos, scale_units="xy", scale=1)
    ax.annotate(r"$\sigma_1$", bottom_right_pos + np.array([-0.18, 0.45]) * scale)

    ax.quiver(*zero_pos, *bottom_left_pos, scale_units="xy", scale=1)
    # ax.annotate(r"$\sigma_3$", bottom_left_pos + np.array([0.1, -0.45]) * scale)
    ax.annotate(r"$\sigma_3$", bottom_left_pos + np.array([0.28, -0.4]) * scale)
    # ax.annotate(r"$\sigma_3$", bottom_left_pos + np.array([-0.18, -0.4]) * scale)


def get_contour_values_per_yield_surface(principal_stress_points, all_eps_values,
                                         mapping_model, points_per_yield_surface=36):
    """
    Get the yield stress values of yield surfaces according to get_yield function
    defined above and passed in plastic strain values, all_eps_values

    If your equation is only outputting one yield stress value, then you will
    have to multiply out the normalizing term so the yield stresses will vary
    """
    original_yield_values = get_yield(principal_stress_points, eps=all_eps_values, mapping_model=mapping_model)
    n_yield_surfaces = (data.shape[0] // points_per_yield_surface)

    contour_values = []
    for yield_fn_i in range(n_yield_surfaces):
        # average yield surface values across each yield surface, shouldn't
        # be needed with a valid yield surface, but can help with stability issues
        contour_level = np.mean(original_yield_values[yield_fn_i::n_yield_surfaces])

        contour_values.append(contour_level)
    return contour_values


if __name__ == "__main__":
    data = np.loadtxt("../data/processed_data/vpsc_57_bingo_format.txt")
    # data = np.loadtxt("../data/processed_data/hill_w_hardening.txt")
    data = data[np.invert(np.all(np.isnan(data), axis=1))]
    points_per_yield_surface = 37

    yield_points_3d = data[:, :3]
    all_eps = data[:, 3][:, None, None]
    yield_points_pi_plane = (yield_points_3d @ align_pi_plane_with_axes_rot())[:, :2]

    coord_range = [-600, 600]
    model = vpsc_57_mapping_model

    # coord_range = [-60, 60]
    # model = hill_mapping_model

    mappings = model(all_eps)
    mapped_points_3d = (mappings @ yield_points_3d[:, :, None]).squeeze()
    mapped_points_pi_plane = (mapped_points_3d @ align_pi_plane_with_axes_rot())[:, :2]

    fig = plt.figure()
    ax = fig.add_subplot()

    def plot_yield_surfaces():
        yield_surface_contours = np.array(get_contour_values_per_yield_surface(yield_points_3d.reshape((-1, 3, 1)), all_eps, model, points_per_yield_surface))
        print("yield surface contour levels:", yield_surface_contours)
        print()
        for i, eps_to_plot in enumerate(np.unique(all_eps)):
            level = yield_surface_contours[i]

            yield_fn = partial(get_yield, eps=eps_to_plot, mapping_model=model)
            fn_yield_points, _ = get_points_from_yield(yield_fn, coord_range,
                                                       level_to_plot=level,
                                                       coord_n=100)
            print(f"n of points from yield contour {i}:", len(fn_yield_points))
        print()

    def plot_real_points():
        return ax.scatter(*yield_points_pi_plane.T, s=5, label="actual")

    def plot_mapped_points():
        return ax.scatter(*mapped_points_pi_plane.T, s=5, label="mapped")

    plot_yield_surfaces()
    real_handle = plot_real_points()

    average_scale_between_mapped_and_actual = np.median(np.mean(yield_points_pi_plane / mapped_points_pi_plane, axis=1))
    print("average scale between mapped and actual points:", average_scale_between_mapped_and_actual)
    if np.abs(average_scale_between_mapped_and_actual) > 10 or np.abs(average_scale_between_mapped_and_actual) < 0.1:
        print(f"\t\033[91mWarning: Average scale between average and mapped points is high")
        print("\tTo fix this, scale the mapping individual by the average shown above")
        print("\t ... automatically scaling by points by average")
        mapped_points_pi_plane *= average_scale_between_mapped_and_actual

    mapped_handle = plot_mapped_points()

    draw_principal_axes(ax, length_of_axes=550)

    # draw_principal_axes(ax, length_of_axes=42, scale=10)

    # coord_range = [-60, 60]
    ax.axis("equal")
    ax.set_xlim(*coord_range)
    ax.set_ylim(*coord_range)

    yield_surface_legend_rectangle = plt.Rectangle((0, 0), 1, 1, fc="purple")
    # ax.legend(handles=[real_handle, mapped_handle, yield_surface_legend_rectangle], labels=["yield points", "mapped points", "found yield surface"])
    ax.legend(handles=[real_handle, mapped_handle], labels=["yield points", "mapped points"])
    ax.set_xlabel("$S_1$ (MPa)")
    ax.set_ylabel("$S_2$ (MPa)")

    plt.show()
