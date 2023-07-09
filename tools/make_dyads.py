def make_dyads(theta_samples, phi_samples):
    """
    Args:
        theta_samples:
        phi_samples:
    """
    import numpy as np
    import scipy.linalg as la
    v = np.array([np.sin(theta_samples) * np.cos(phi_samples), np.sin(theta_samples) * np.sin(phi_samples), np.cos(theta_samples)])
    dyadic_tensor = np.dot(v, v.T)
    dyadic_tensor = dyadic_tensor / len(theta_samples)
    L, E = la.eig(dyadic_tensor)
    ind = np.argsort(-L, kind='quicksort')
    v1 = np.transpose(E[:, ind[0]])
    disp = 1 - np.max(np.max(np.abs(L)))
    return v1, disp


def make_dyads_cart(Vsamples):
    """
    Args:
        Vsamples:
    """
    import numpy as np
    import scipy.linalg as la
    N = Vsamples.shape[0]
    dyadic_tensor = 0;
    for l in range(0, N):
        v = np.transpose(Vsamples[l, :])
        dyadic_tensor = dyadic_tensor + v * np.transpose(v)
    dyadic_tensor = dyadic_tensor / N
    E, L = la.eig(dyadic_tensor)  # equivalent of Matlab's eig(dyadic_tensor)
    ind = np.argsort(-np.diag(L), kind='quicksort')  # equivalent to Matlab sort(diag(L),'descend')
    v1 = np.transpose(E[:, ind[0]])
    disp = 1 - np.max(np.max(np.abs(L)))
    return v1, disp


