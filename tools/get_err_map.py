def get_error(im1, im2, type_error):
    """
    Args:
        im1:
        im2:
        type_error:
    """
    import numpy as np
    if (type_error == 1):
        err = im1 - im2
    elif (type_error == 2):
        err = 100 * (im1 - im2)/np.max(im1, im2)

    return err

def get_vec_error(v1, v2, type_error):
    """
    Args:
        v1:
        v2:
        type_error:
    """
    import numpy as np
    import math
    if ((type_error == 1) or (type_error == 'dot_product')):
        err = v1@v2
    elif ((type_error == 2) or (type_error == 'cross_angle_degrees')):
        err = np.round(math.degrees(np.arccos(v1@v2)))
        if err > 90:  # Reduction to first quadrant
            err = 180 - err
    elif ((type_error == 3) or (type_error == 'cross_orientation')):
        pass

    return err

def get_err_vector_map(im1, im2, mask=None, mod_im1=None, mod_im2=None, type_error='cross_angle'):
    """
    Args:
        im1:
        im2:
        mask: preferrably, use a FA mask
        mod_im1:
        mod_im2:
        type_error: 'dot_product', 'cross_angle', 'cross_orientation'
    """
    import numpy as np
    from tools.get_err_map import get_vec_error
    if (im1.shape != im2.shape):
        print('Error! Vector maps don\'t have same dimensionality.')
        #break
    else:
        if ((len(im1.shape)==1) or (len(im1.shape)==2)):
            # It is a single voxel
            error = get_vec_error(im1, im2, type_error)
        elif (len(im1.shape)==3):
            # It is a bunch of voxels or slice - 2D map
            error = np.array([get_vec_error(im1[x, y], im2[x, y], type_error) \
                                  if mask[x, y] != 0 else 0 \
                              for x in range(0, im1.shape[0]) \
                              for y in range(0, im1.shape[1])]).reshape(im1.shape)
        elif (len(im1.shape)==4):
            # It is a 3D Maps
            #### CHECK dotprod_3d function!
            error = np.array([get_vec_error(im1[x, y, z], im2[x, y, z], type_error) \
                                  if mask[x, y, z] != 0 else 0 \
                              for x in range(0, im1.shape[0]) \
                              for y in range(0, im1.shape[1]) \
                              for z in range(0, im1.shape[2])]).reshape(im1.shape)

        return error


def get_err_scalar_map(im1, im2, mask=None, type_error=2):
    """
    Args:
        im1:
        im2:
        mask: preferrably, use a FA mask
        type_error: 1=percentage, 2=absolute, 3=absolute normalized
    """
    import numpy as np
    from tools.get_err_map import get_error
    if (im1.shape != im2.shape):
        print('Error! Scalar maps don\'t have same dimensionality.')
        # break
    else:
        if (len(im1.shape)==1):
            # It is a single voxel
            error = get_error(im1, im2, type_error)
        elif (len(im1.shape)==2):
            # It is a bunch of voxels or slice - 2D map
            error = np.array([ get_error(im1[x,y],im2[x,y], type_error) \
                                   if mask[x,y]!=0 else 0 \
                               for x in range(0,im1.shape[0]) \
                               for y in range(0,im1.shape[1]) ]).reshape(im1.shape)
        elif (len(im1.shape)==3):
            # It is a 3D Map
            error = np.array([ get_error(im1[x,y,z],im2[x,y,z], type_error) \
                                   if mask[x,y,z]!=0 else 0 \
                               for x in range(0,im1.shape[0]) \
                               for y in range(0,im1.shape[1]) \
                               for z in range(0,im1.shape[2]) ]).reshape(im1.shape)

        return error

