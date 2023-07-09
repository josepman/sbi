from numpy import linalg as la
import numpy as np

def nearestPD(A):
    """Find the nearest positive-definite matrix to input A Python/Numpy port of
    John D'Errico's `nearestSPD` MATLAB code [1], which credits [2]. [1]
    https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd [2]
    N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix"
    (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    Args:
        A:
    """
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3
    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky

    Args:
        B:
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def run_nearestSPD(A, mask=None):
    """For cases when A is an array of cov. matrices

    Args:
        A:
        mask:
    """
    n = len(A.shape) - 2    # How many dimensions (-2 because cov is a square matrix for each voxel)
    covSPD = A.copy()
    if mask is not None:
        if n==2:
            for x in range(0, A.shape[0]):
                for y in range(0, A.shape[1]):
                    if (mask[x,y] > 0 and isPD(A[x,y])==False):
                        covSPD[x,y] = nearestPD(A[x,y])
        if n==3:
            for x in range(0, A.shape[0]):
                for y in range(0, A.shape[1]):
                    for z in range(0, A.shape[2]):
                        if (mask[x, y, z] > 0 and isPD(A[x,y,z])==False):
                            covSPD[x,y,z] = nearestPD(A[x,y,z])

    else:
        if n==2:
            for x in range(0, A.shape[0]):
                for y in range(0, A.shape[1]):
                    if isPD(A[x, y]) == False:
                        covSPD[x,y] = nearestPD(A[x,y])
        if n==3:
            for x in range(0, A.shape[0]):
                for y in range(0, A.shape[1]):
                    for z in range(0, A.shape[2]):
                        if isPD(A[x,y,z])==False:
                            covSPD[x,y,z] = nearestPD(A[x,y,z])

    return covSPD
