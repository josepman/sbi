def xangle2v(x_angle, v1=None, output='sph'):
    """
    Args:
        x_angle:
        v1:
        output:
    """
    #E.g.:
    #x_angles = np.linspace(np.pi / 6, np.pi / 2, 16)  # From 30 to 90 degrees, in steps of 4 deg. Use math.degrees(x_angles) to convert this radian-vector in degrees-vector

    import numpy as np
    from tools.cart2sph import cart2sph
    # If v1 is not provided, it will be created randomly
    if v1 is None:
        th1 = np.random.uniform(0, np.pi / 2, len(x_angle))
        ph1 = np.random.uniform(0, np.pi, len(x_angle))
        v1 = np.array([np.sin(th1) * np.cos(ph1), np.sin(th1) * np.sin(ph1), np.cos(th1)])
        mod1 = np.sqrt(np.square(v1[0, :]) + np.square(v1[1, :]) + np.square(v1[2, :]))
        v1 = v1 / mod1  # make it unitary
        th1, ph1 = cart2sph(v1)

    # Having v1 and given the crossing_angle, v2 can be just estimated adding the angle to one of the sph components of v1:
    th2 = th1 + x_angle
    ph2 = ph1.copy()
    
    if output=='sph':
        return th1,ph1,th2,ph2
    else:
        v2 = np.array([np.sin(th2) * np.cos(ph2), np.sin(th2) * np.sin(ph2), np.cos(th2)])
        mod2 = np.sqrt(np.square(v2[0, :]) + np.square(v2[1, :]) + np.square(v2[2,:]))
        v2 = v2 / mod2
        return v1, v2