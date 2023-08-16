import numpy as np
import pandas as pd
import dill


def P_indv(eps):
    X_0 = eps
    from numpy import array, sin, cos

    P = cos(cos(((sin((array([[-0.67662144,  0.7613563 ],
                              [ 0.26741559, -0.59678159]]))*(X_0)))*(X_0 + X_0))@(array([[-631.30903812, -153.5812359 ],
                                                                                         [-386.65304506, -265.4762436 ]])) + array([[-0.67662144,  0.7613563 ],
                                                                                                                                    [ 0.26741559, -0.59678159]])) + array([[ 1.24679185, -1.53826878],
                                                                                                                                                                           [ 1.84911831,  1.16704097]]) + array([[-0.67662144,  0.7613563 ],
                                                                                                                                                                                                                 [ 0.26741559, -0.59678159]]))

    return P


if __name__ == "__main__":
    original_name = "vpsc_evo_17_data_2d_transpose_implicit_format"
    data = np.loadtxt(original_name + ".txt")
    data = data[np.invert(np.all(np.isnan(data), axis=1))]

    eps_index = data.shape[1] - 1

    eps = data[:, eps_index][:, None, None]

    P_actual = P_indv(eps)
    P_vm = np.array([[1, -0.5, -0.5],
                     [-0.5, 1, -0.5],
                     [-0.5, -0.5, 1]]) / 100

    P_vm = np.array([[1, -0.5],
                     [-0.5, 1]]) / 100

    P_vm = np.repeat(P_vm[None, :, :], repeats=len(P_actual), axis=0)

    df = pd.DataFrame(data={"eps": eps.squeeze().tolist(), "P_actual": P_actual.tolist(), "P_vm": P_vm.tolist()})

    df.to_pickle(original_name + "_mapping_df.pkl")
    print("done. saved to", original_name + "_mapping_df.pkl")
