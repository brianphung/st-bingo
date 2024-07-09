from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from research.utility.rotations import align_pi_plane_with_axes_rot
import dill

def mapping_model(eps, checkpoint_object, individual=-1):

    eq = checkpoint_object.hall_of_fame[individual].evaluate_equation_at(eps)

    scaling_factor = 1.0
    # set scaling factor if needed (e.g., if average scale is too large)
    eq *= scaling_factor

    return eq


def get_yield(principal_stress, *, eps=0.0, mapping_model, checkpoint_object, individual=-1):
    """
    Get real yield stress values as a function of plastic strain using the
    mapping provided above
    """
    if len(principal_stress.shape) == 2:
        principal_stress = principal_stress[None, :, :]

    if isinstance(eps, np.ndarray) and eps.ndim == 2:
        eps = np.expand_dims(eps, 0)

    mapping_matrix = mapping_model(eps, checkpoint_object, individual=-1)

    P_fict = np.array([[1, -0.5, -0.5, 0, 0, 0],
                       [-0.5, 1, -0.5, 0, 0, 0],
                       [-0.5, -0.5, 1, 0, 0, 0],
                       [0, 0, 0, 3, 0, 0],
                       [0, 0, 0, 0, 3, 0],
                       [0, 0, 0, 0, 0, 3]])
    print('map mat', mapping_matrix)
    P_real = mapping_matrix.transpose((0, 2, 1)) @ P_fict @ mapping_matrix
    print('principal stress', principal_stress.transpose((0, 2, 1)))
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


def get_contour_values_per_yield_surface(principal_stress_points, all_eps_values,
                                         mapping_model, n_yield_surfaces,
                                         checkpoint_object, individual):
    """
    Get the yield stress values of yield surfaces according to get_yield function
    defined above and passed in plastic strain values, all_eps_values

    If your equation is only outputting one yield stress value, then you will
    have to multiply out the normalizing term so the yield stresses will vary
    """
    original_yield_values = get_yield(principal_stress_points, eps=all_eps_values, mapping_model=mapping_model,
                                      checkpoint_object=checkpoint_object, individual=individual)

    contour_values = []
    for yield_fn_i in range(n_yield_surfaces):
        # average yield surface values across each yield surface, shouldn't
        # be needed with a valid yield surface, but can help with stability issues
        contour_level = np.mean(original_yield_values[yield_fn_i::n_yield_surfaces])

        contour_values.append(contour_level)
    return contour_values


def plot_mapping(formatted_data_path,
                 checkpoint_file,
                 range_to_sample_contours,
                 *,
                 plot_original_points, plot_mapped_points,
                 plot_yield_surfaces,
                 drawn_axes_length=None, drawn_axes_scale=None,
                 figure_range=None, comps=[0,1], individual=-1):
    """
    Plots the yield points provided from formatted data path, their
    mapped versions according to the provided mapping model, and the
    mapped yield surface implied by the provided mapping model.
    :param formatted_data_path: Data path of the yield surface data
    formatted for bingo.
    :param mapping_model: Model that outputs mapping matrices based on a
    provided state parameter
    :param range_to_sample_contours: Coordinate range in which to sample
    the yield contours/surfaces
    :param drawn_axes_length: The length of the drawn principal stress
    axes
    :param drawn_axes_scale: A quantity that adjusts how far the $sigma_i$
    symbols are away from the drawn principal stress axes
    :param plot_original_points: Whether to plot the original yield points
    :param plot_mapped_points: Whether to plot the mapped yield points
    :param plot_yield_surfaces: Whether to plot the implied yield
    contours/surfaces or not
    :param figure_range: The range of the figure's axes
    """
    if figure_range is None:
        figure_range = range_to_sample_contours

    comp_map = [ '$\\sigma_{11}', '$\\sigma_{22}', '$\\sigma_{33}', '$\\tau_{23}', '$\\tau_{13}', '$\\tau_{12}' ]

    # load data
    data = np.loadtxt(formatted_data_path)
    nan_rows = np.all(np.isnan(data), axis=1)
    n_yield_surfaces = np.argmax(nan_rows)
    data = data[np.invert(nan_rows)]

    print("n of yield surfaces:", n_yield_surfaces)

    # get yield points and eps values
    yield_points_3d = data[:, :6]

    all_eps = data[:, 6][:, None, None]
    #yield_points_pi_plane = (yield_points_3d @ align_pi_plane_with_axes_rot())[:, :2]

    checkpoint_buffer = open(checkpoint_file, 'rb')
    checkpoint_object = dill.load(checkpoint_buffer)

    # get mapped yield points
    mappings = mapping_model(all_eps, checkpoint_object)
    mapped_points_3d = (mappings @ yield_points_3d[:, :, None]).squeeze()
    #mapped_points_pi_plane = (mapped_points_3d @ align_pi_plane_with_axes_rot())[:, :2]

    fig = plt.figure()
    ax = fig.add_subplot()

    def _plot_yield_surfaces():
        # get yield contour values as determined by mapping_model
        print('yP 3d', yield_points_3d)
        print('reshaped', yield_points_3d.reshape((-1, 6, 1)))
        yield_surface_contours = get_contour_values_per_yield_surface(
            yield_points_3d.reshape((-1, 6, 1)),
            all_eps,
            mapping_model,
            n_yield_surfaces,
            checkpoint_object, individual
        )
        yield_surface_contours = np.array(yield_surface_contours)
        print("yield surface contour levels:", yield_surface_contours)

        # plot each yield surface contour
        for i, eps_to_plot in enumerate(np.unique(all_eps)):
            level = yield_surface_contours[i]

            yield_fn = partial(
                get_yield,
                eps=eps_to_plot,
                mapping_model=mapping_model
            )
            # fn_yield_points, _ = get_points_from_yield(
            #     yield_fn,
            #     range_to_sample_contours,
            #     level_to_plot=level,
            #     coord_n=100
            # )
            # print(f"n of points from yield contour {i}:", len(fn_yield_points))

    def _plot_real_points():
        return ax.scatter(yield_points_3d[:,comps[0]], yield_points_3d[:, comps[1]], s=5, label="actual")

    def _plot_mapped_points():
        return ax.scatter(mapped_points_3d[:,comps[0]], mapped_points_3d[:,comps[1]], s=5, label="mapped")

    legend_handles = []
    legend_labels = []
    if plot_original_points:
        real_handle = _plot_real_points()
        legend_handles.append(real_handle)
        legend_labels.append("yield points")

    if plot_mapped_points:
        # determine average scale between mapped and actual yield points,
        # print error message if too large or small
        average_scale_between_mapped_and_actual = 1.0
        print("average scale between mapped and actual points:", average_scale_between_mapped_and_actual)
        if np.abs(average_scale_between_mapped_and_actual) > 10 or np.abs(average_scale_between_mapped_and_actual) < 0.1:
            print(f"\t\033[91mWarning: Average scale between average and mapped points is high")
            print("\tTo fix this, scale the mapping individual by the average shown above")
            print("\t ... automatically scaling by points by average\033[0m")
            mapped_points_pi_plane *= average_scale_between_mapped_and_actual

        mapped_handle = _plot_mapped_points()
        legend_handles.append(mapped_handle)
        legend_labels.append("mapped points")

    if plot_yield_surfaces:
        _plot_yield_surfaces()
        yield_surface_legend_rectangle = plt.Rectangle((0, 0), 1, 1, fc="purple")
        legend_handles.append(yield_surface_legend_rectangle)
        legend_labels.append("found yield surface")

    #draw_principal_axes(ax, length_of_axes=drawn_axes_length, scale=drawn_axes_scale)

    ax.axis("equal")
    ax.set_xlim(*figure_range)
    ax.set_ylim(*figure_range)

    ax.legend(handles=legend_handles, labels=legend_labels)
    ax.set_xlabel("$\\sigma_{11}$ (MPa)")
    ax.set_ylabel("$\\sigma_{22}$ (MPa)")

    plt.show()


if __name__ == "__main__":

    training_data_path = '../data_6x6/processed_data'
    hill_data_path = f"{training_data_path}/Hill_bingo_format.txt"
    checkpoint_file = "../experiments/checkpoints/hill6x6/checkpoint_101.pkl"

    print("plotting results:")
    # plot hill mapped points
    plot_mapping(hill_data_path, checkpoint_file, [-20, 20],
                 plot_original_points=True, plot_mapped_points=True, plot_yield_surfaces=False,
                 drawn_axes_length=42, drawn_axes_scale=10, figure_range=[-10, 10],
                 comps=[0, 1])
    print()

    # # plot hill mapped yield surface
    # plot_mapping(hill_data_path, checkpoint_file, [-40, 40],
    #              plot_original_points=True, plot_mapped_points=False, plot_yield_surfaces=True,
    #              drawn_axes_length=42, drawn_axes_scale=10, figure_range=[-60, 60])
