import numpy as np
import pandas as pd


if __name__ == '__main__':
    P_actual = np.array([[-0.00882189,  0.01795338, -0.00398629],
       [-0.00903886, -0.00944096,  0.01903115],
       [ 0.01370361, -0.00956715, -0.00983265]])
    P_desired = np.array([[1, -0.5, -0.5],
                     [-0.5, 1, -0.5],
                     [-0.5, -0.5, 1]]) / 100
    print(P_desired)

    # make P_actual symmetric
    P_actual = (P_actual + P_actual.T) / 2
    print(repr(P_actual))


    def get_loss(mapping_mat_flattened, P_in, P_out, minimize=False):
        mapping_mat = mapping_mat_flattened.reshape((3, 3))

        P_pred = mapping_mat.T @ P_in @ mapping_mat
        # P_pred = mapping_mat @ P_in
        loss = (P_pred - P_out).flatten()
        loss /= P_out.flatten()
        if minimize:
            loss = np.mean(np.square(loss))
        return loss


    def jac(mapping_mat_flattened, P_in, P_out, minimize=False):
        mapping_mat = mapping_mat_flattened.reshape((3, 3))
        P_pred = (mapping_mat @ P_in).flatten()
        n = len(mapping_mat_flattened)
        if minimize:
            grad = 2.0 / n * P_in.flatten() * (P_pred - P_out.flatten())
            return grad / P_out.flatten()
        else:
            return np.diag(P_in.flatten() / P_out.flatten())
            # return np.repeat(P_in.reshape((n, 1)), n, axis=1).T


    from scipy.optimize import root, minimize

    P_input = P_desired
    P_output = P_actual

    # P_input = P_actual
    # P_output = P_desired

    # optim_result = root(get_loss, x0=np.random.randn(9), args=(P_input, P_output, False), jac=jac, method="lm")
    optim_result = root(get_loss, x0=np.random.randn(9), args=(P_input, P_output), method="lm")

    # optim_result = minimize(get_loss, x0=np.ones(9), args=(P_input, P_output, True), method="SLSQP", jac=jac, options={"eps": 1e-7})
    # optim_result = minimize(get_loss, x0=np.ones(9), args=(P_input, P_output, True), method="BFGS", options={"gtol": 1e-15, "eps": 1e-5, "xrtol": 1e-15})

    # optim_result = minimize(get_loss, x0=np.ones(9), args=(P_input, P_output, True), jac=jac, method="BFGS", options={"gtol": 1e-15, "eps": 1e-5})
    # optim_result = minimize(get_loss, x0=np.ones(9), args=(P_input, P_output, True), method="BFGS")

    found_mapping_mat = optim_result.x.reshape((3, 3))
    found_mapping_mat = np.array([[-0.16440424, -0.63960819,  0.39540635],
                               [-0.23151857, -0.70180533,  0.37946872],
                               [-0.2893659 , -0.75541446,  0.36573173]])
    # found_mapping_mat = P_output @ np.linalg.pinv(P_input)
    # found_mapping_mat = np.linalg.inv(found_mapping_mat)
    print(optim_result)
    print("\nfound result:", repr(found_mapping_mat))
    print("loss of found matrix:", get_loss(found_mapping_mat, P_actual, P_desired))
    print("\tmse:", get_loss(found_mapping_mat, P_actual, P_desired, True))

    print("\n output mat:", P_output)
    # print("\n mapped input mat:", found_mapping_mat @ P_input)
    print("\n mapped input mat:", found_mapping_mat.T @ P_input @ found_mapping_mat)
