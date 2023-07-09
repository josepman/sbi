def v2xangle(v1,v2):
    """v1 and v2 have to be in cartesian coords

    Args:
        v1:
        v2:
    """

    '''th and ph can be a vector of values. Can it be a map??'''
    import numpy as np
    from tools.sph2cart import sph2cart
    v1 = [sph2cart(th1[i], ph1[i]) for i in range(0, len(th1))]
    v2 = [sph2cart(th2[i], ph2[i]) for i in range(0, len(th2))]

    # Alternative way:
    # v1 = np.array([np.sin(th1) * np.cos(ph1), np.sin(th1) * np.sin(ph1), np.cos(th1)])
    # mod_v1_v = np.sqrt(np.square(v1[0, :]) + np.square(v1[1, :]) + np.square(v1[2, :]))  # np.sqrt(np.square(
    # np.cos(ph1_v)) * np.square(np.sin(th1_v)) + np.square(np.sin(th1_v)) * np.square(np.sin(ph1_v)) + np.square(
    # np.cos(th1_v)))
    # v1 = v1 / mod_v1_v  # make it unitary
    return x_angle