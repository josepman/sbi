def sph2cart(theta, phi):
    """
    Args:
        theta:
        phi:
    """
    import numpy as np
    # theta = -2*np.pi + theta + np.pi/2

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    v = np.array([x, y, z])

    return v  # sign '-' because of the change of theta
