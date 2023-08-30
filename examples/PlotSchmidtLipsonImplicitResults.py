import numpy as np

from VPSCWithHardeningParentAgraphImplicitExample import get_dx_dt
import matplotlib.pyplot as plt


def generate_circle_data(n_points):
    t = np.linspace(0, 2 * np.pi, num=n_points)
    points = np.hstack((np.cos(t)[:, None], np.sin(t)[:, None]))
    return points


def found_equation(x, y):
    X_0 = x
    X_1 = y
    eq_str = "(0.03171195982827554)((X_0)(X_0) + (X_1)(X_1))"
    return eval(eq_str.replace(")(", ")*("))


def get_result_contour_points():
    coord_ranges = [-1, 1]
    coord_n = 100
    x = np.linspace(*coord_ranges, num=coord_n)
    y = np.linspace(*coord_ranges, num=coord_n)
    xx, yy = np.meshgrid(x, y)

    h = np.zeros((coord_n, coord_n))
    for i in range(coord_n):
        for j in range(coord_n):
            h[i][j] = found_equation(xx[i][j], yy[i][j])
    cs = plt.contour(x, y, h, levels=[0.03])
    return np.array([])
    # return np.concatenate(np.array(cs.collections[0].get_paths()))


if __name__ == "__main__":
    x = generate_circle_data(100)
    dx_dt = get_dx_dt(x)

    contour_levels = np.array([found_equation(*pair) for pair in x])
    print(contour_levels)

    get_result_contour_points()

    # plt.scatter(*x.T)
    # plt.quiver(*x.T, *dx_dt.T, scale=1, scale_units="xy")
    plt.show()

