def get_unc_cones(th_samples, ph_samples, dyads=None, perc=100):
    import numpy as np
    import math
    from tools.make_dyads import make_dyads
    from tools.sph2cart import sph2cart
    angles = np.zeros((len(th_samples)))
    v = sph2cart(th_samples, ph_samples)
    if dyads is None:
        dyads, _ = make_dyads(th_samples, ph_samples)

    for i in range(0, len(th_samples)):
        angles[i] = math.degrees(np.arccos(np.dot(dyads, v[:,i])))
        if angles[i]>90:
            angles[i] = angles[i]-90

    angles = np.sort(angles)
    if perc is not None:
        angle_CI = np.percentile(angles, perc)
    else:
        angle_CI = np.max(angles)

    return angle_CI

def get_unc_cones_map(th_samples, ph_samples, mask=None, CI=95, output_file=None, aff_mat=None):
    import numpy as np
    import nibabel as nb
    #3D MAP
    if (len(th_samples.shape)==4):
        angle_CI = np.array([get_unc_cones(th_samples[x, y, z], ph_samples[x, y, z], perc=CI) \
                              if mask[x, y, z] != 0 else 0 \
                          for x in range(0, th_samples.shape[0]) \
                          for y in range(0, th_samples.shape[1]) \
                          for z in range(0, th_samples.shape[2])]).reshape(th_samples.shape)

    # SLICE
    elif (len(th_samples.shape)==3):
        angle_CI = np.array([get_unc_cones(th_samples[x, y], ph_samples[x, y], perc=CI) \
                                 if mask[x, y] != 0 else 0 \
                             for x in range(0, th_samples.shape[0]) \
                             for y in range(0, th_samples.shape[1])]).reshape(th_samples.shape)
    # ARRAY
    elif (len(th_samples.shape)==3):
        angle_CI = np.array([get_unc_cones(th_samples[x], ph_samples[x], perc=CI) \
                                 if mask[x] != 0 else 0 \
                             for x in range(0, th_samples.shape[0])]).reshape(th_samples.shape)

    if output_file:
        if aff_mat:
            nb.save(nb.Nifti2Image(angle_CI, affine=aff_mat), f'{output_file}unc_map_{CI}.nii.gz')
        else:
            np.save(f'{output_file}unc_map_{CI}.npy', angle_CI)

    return angle_CI