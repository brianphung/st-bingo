from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from research.utility.rotations import align_axes_with_pi_plane_rot, align_pi_plane_with_axes_rot
import dill
import sys

def vm_1_mapping_model(eps):
    from numpy import sin, cos, array

    # bingo uses sqrt(|x|) as sqrt(x)
    def sqrt(x):
        return np.sqrt(np.abs(x))

    X_0 = eps
    if not isinstance(eps, np.ndarray):
        eps = np.array([eps])[:, None, None]

    # input your individual here
    x = array([ -0.6643076935208633, -12.078321700379332 ,   0.3445924187120184,
        -2.241075086881452 ,  -1.7312605745140484,   0.8563298787383459,
        -1.937376145802561 ,  -1.5232220555666125, 310.00248804012773  ,
         3.1313382630315925,  11.92019618        ,  22.12241255        ,
        67.75485734        ,  11.34285823        ,  22.7518725         ,
        68.42081951        ,  11.30870116        ,  22.22108227        ])
    eq = (x[1]*X_0 + x[0])  * (array([[x[2], x[3], x[4] ],  
        [x[3] , x[6], x[5]],
        [x[4], x[5], x[7]]]))/((x[8]*X_0 + x[9]) )
    x = array([-0.324190879399528  , -0.5910265708699819 , -0.18063603062696332,
       -0.38121381734569654, -0.6175628700881645 , -0.23576228096254312,
       -0.28240916917730036, -0.5473481732181913 , -0.16744427128767322,
       -0.2630318047470945 , -0.4235775835834732 , -0.04148242995369289,
        0.5943025397254625 ,  0.5770764819925227 ,  0.05778373947936392,
       -0.4019021627391064 ,  0.6422532886726127 , -0.9725103257468183 ,
        0.564662540091747  , -0.32419465         , -0.59102337         ,
       -0.18063179         , -0.38121066         , -0.61755865         ,
       -0.23576606         , -0.28240491         , -0.54735189         ,
       -0.16744104         , -0.358838           , -0.42357774         ,
       -0.04148228         ,  0.5943027          ,  0.57707663         ,
        0.0577839          , -0.40190232         ,  0.64225345         ,
       -0.97251048         ,  0.56466269         ])
    eq = (array([[x[0], x[1], x[2]],
            [x[3], x[4], x[5]],
            [x[6], x[7], x[8]]])) * ((array(x[9]) + X_0 + array(x[9]) + X_0) * (X_0)) + array([[x[10], x[11], x[12] ],        
            [ x[13],  x[14] , x[15]],
            [ x[16], x[17],  x[18]]])
    
    # x = array([-3.1583012866766879e-08,  1.0000000053393843e+00])
    # eq = ((x[0]*X_0 + x[1]) ) * np.eye(3)

    scaling_factor = 1.0
    # set scaling factor if needed (e.g., if average scale is too large)
    eq *= scaling_factor

    if eq.shape[0] != eps.shape[0]:
        eq = np.repeat(eq[None, :, :], eps.shape[0], axis=0)

    return eq

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



def vpsc_75_mapping_model(eps):
    from numpy import sin, cos, array

    if 'darwin' in sys.platform:
        individual_path = "/Users/brian/Work/software/st-bingo/research/experiments/checkpoints/vpsc75/checkpoint_501.pkl"
    else:
        individual_path = None
    with open(individual_path, 'rb') as f:
        model = dill.load(f)

    eps = np.array([eps[0,0,:]])
    eq = model.hall_of_fame[0].evaluate_equation_at(eps)[0]
    eq *= (100 + 1050*eps)
    eq = align_axes_with_pi_plane_rot().T @ eq @ align_axes_with_pi_plane_rot()
    eq[:, 2] = 0
    eq[2, :] = 0
    eq[2, 2] = 1
    eq = align_axes_with_pi_plane_rot() @ eq @ align_axes_with_pi_plane_rot().T

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
                                         mapping_model, n_yield_surfaces):
    """
    Get the yield stress values of yield surfaces according to get_yield function
    defined above and passed in plastic strain values, all_eps_values

    If your equation is only outputting one yield stress value, then you will
    have to multiply out the normalizing term so the yield stresses will vary
    """
    original_yield_values = get_yield(principal_stress_points, eps=all_eps_values, mapping_model=mapping_model)

    contour_values = []
    for yield_fn_i in range(n_yield_surfaces):
        # average yield surface values across each yield surface, shouldn't
        # be needed with a valid yield surface, but can help with stability issues
        contour_level = np.mean(original_yield_values[yield_fn_i::n_yield_surfaces])

        contour_values.append(contour_level)
    return contour_values


def plot_mapping(formatted_data_path, mapping_model,
                 range_to_sample_contours,
                 *,
                 plot_original_points, plot_mapped_points,
                 plot_yield_surfaces,
                 drawn_axes_length=None, drawn_axes_scale=None,
                 figure_range=None):
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

    # load data
    data = np.loadtxt(formatted_data_path)
    nan_rows = np.all(np.isnan(data), axis=1)
    n_yield_surfaces = np.argmax(nan_rows)
    data = data[np.invert(nan_rows)]

    print("n of yield surfaces:", n_yield_surfaces)

    # get yield points and eps values
    yield_points_3d = data[:, :3]
    all_eps = data[:, 3][:, None, None]
    yield_points_pi_plane = (yield_points_3d @ align_pi_plane_with_axes_rot())[:, :2]

    # get mapped yield points
    mappings = mapping_model(all_eps)
    mapped_points_3d = (mappings @ yield_points_3d[:, :, None]).squeeze()
    mapped_points_pi_plane = (mapped_points_3d @ align_pi_plane_with_axes_rot())[:, :2]

    fig = plt.figure()
    ax = fig.add_subplot()

    def _plot_yield_surfaces():
        # get yield contour values as determined by mapping_model
        yield_surface_contours = get_contour_values_per_yield_surface(
            yield_points_3d.reshape((-1, 3, 1)),
            all_eps,
            mapping_model,
            n_yield_surfaces
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
            fn_yield_points, _ = get_points_from_yield(
                yield_fn,
                range_to_sample_contours,
                level_to_plot=level,
                coord_n=100
            )
            # print(f"n of points from yield contour {i}:", len(fn_yield_points))

    def _plot_real_points():
        return ax.scatter(*yield_points_pi_plane.T, s=5, label="actual")

    def _plot_mapped_points():
        return ax.scatter(*mapped_points_pi_plane.T, s=5, label="mapped")

    legend_handles = []
    legend_labels = []
    if plot_original_points:
        real_handle = _plot_real_points()
        legend_handles.append(real_handle)
        legend_labels.append("yield points")

    if plot_mapped_points:
        # determine average scale between mapped and actual yield points,
        # print error message if too large or small
        average_scale_between_mapped_and_actual = np.median(np.mean(yield_points_pi_plane / mapped_points_pi_plane, axis=1))
        print("average scale between mapped and actual points:", average_scale_between_mapped_and_actual)
        if np.abs(average_scale_between_mapped_and_actual) > 10 or np.abs(average_scale_between_mapped_and_actual) < 0.1:
            print(f"\t\033[91mWarning: Average scale between average and mapped points is high")
            print("\tTo fix this, scale the mapping individual by the average shown above")
            print("\t ... automatically scaling by points by average\033[0m")
            #mapped_points_pi_plane *= average_scale_between_mapped_and_actual

        mapped_handle = _plot_mapped_points()
        legend_handles.append(mapped_handle)
        legend_labels.append("mapped points")

    if plot_yield_surfaces:
        _plot_yield_surfaces()
        yield_surface_legend_rectangle = plt.Rectangle((0, 0), 1, 1, fc="purple")
        legend_handles.append(yield_surface_legend_rectangle)
        legend_labels.append("found yield surface")

    draw_principal_axes(ax, length_of_axes=drawn_axes_length, scale=drawn_axes_scale)

    ax.axis("equal")
    ax.set_xlim(*figure_range)
    ax.set_ylim(*figure_range)

    ax.legend(handles=legend_handles, labels=legend_labels)
    ax.set_xlabel("$S_1$ (MPa)")
    ax.set_ylabel("$S_2$ (MPa)")

    plt.show()


if __name__ == "__main__":
    # plot vspc mapped points
    # print("plotting vpsc results:")
    # vpsc_data_path = "../data/processed_data/vpsc_75_bingo_format.txt"
    # plot_mapping(vpsc_data_path, vpsc_75_mapping_model, [-600, 600],
    #              plot_original_points=True, plot_mapped_points=True, plot_yield_surfaces=False,
    #              drawn_axes_length=550, drawn_axes_scale=100)
    # print()

    # plot vpsc mapped yield surface
    # plot_mapping(vpsc_data_path, vpsc_75_mapping_model, [-600, 600],
    #              plot_original_points=True, plot_mapped_points=False, plot_yield_surfaces=True,
    #              drawn_axes_length=550, drawn_axes_scale=100)
    # print()

    print("plotting vpsc results:")
    vpsc_data_path = "../data/processed_data/vpsc_57_bingo_format.txt"
    plot_mapping(vpsc_data_path, vpsc_57_mapping_model, [-600, 600],
                 plot_original_points=True, plot_mapped_points=True, plot_yield_surfaces=False,
                 drawn_axes_length=550, drawn_axes_scale=100)
    print()
    
    # plot vpsc mapped yield surface
    plot_mapping(vpsc_data_path, vpsc_57_mapping_model, [-600, 600],
                 plot_original_points=True, plot_mapped_points=False, plot_yield_surfaces=True,
                 drawn_axes_length=550, drawn_axes_scale=100)
    print()

    # plot vpsc mapped yield surface
    # plot_mapping(vpsc_data_path, vpsc_75_mapping_model, [-600, 600],
    #              plot_original_points=True, plot_mapped_points=False, plot_yield_surfaces=True,
    #              drawn_axes_length=550, drawn_axes_scale=100)
    # print()

    # print("plotting hill results:")
    # # plot hill mapped points
    # hill_data_path = "../data/processed_data/hill_w_hardening.txt"
    # plot_mapping(hill_data_path, hill_mapping_model, [-40, 40],
    #              plot_original_points=True, plot_mapped_points=True, plot_yield_surfaces=False,
    #              drawn_axes_length=42, drawn_axes_scale=10, figure_range=[-60, 60])
    # print()

    # # plot hill mapped yield surface
    # plot_mapping(hill_data_path, hill_mapping_model, [-40, 40],
    #              plot_original_points=True, plot_mapped_points=False, plot_yield_surfaces=True,
    #              drawn_axes_length=42, drawn_axes_scale=10, figure_range=[-60, 60])

    """
    print("plotting vm results:")
    # plot hill mapped points
    hill_data_path = "../data/processed_data/vm_1_bingo_format.txt"
    plot_mapping(hill_data_path, vm_1_mapping_model, [-40, 40],
                 plot_original_points=True, plot_mapped_points=True, plot_yield_surfaces=False,
                 drawn_axes_length=200, drawn_axes_scale=10, figure_range=[-300, 300])
    print()

    # plot hill mapped yield surface
    plot_mapping(hill_data_path, vm_1_mapping_model, [-40, 40],
                 plot_original_points=True, plot_mapped_points=False, plot_yield_surfaces=True,
                 drawn_axes_length=200, drawn_axes_scale=10, figure_range=[-300, 300])
    """