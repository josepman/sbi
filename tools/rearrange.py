def rearrange(data, dti_v1, bvecs, mask, out_path, name='default', njobs=1):
    import numpy as np
    from joblib import Parallel, delayed
    import os
    from tools.prepare_NIFTI import export_nifti

    # data_rearranged = np.memmap(os.path.join(out_path, 'data_rearranged_memmap'), mode='w+', shape=data.shape)
    #
    # def rearrange_voxel(data, data_rearranged, dti_v1, bvecs, coords):
    #     x,y,z = coords
    #     b0_idx = [0, 17, 33, 35, 52, 68]  # Filter b0 values
    #     data_aux = np.delete(data[x, y, z], b0_idx)
    #     bvecs_aux = np.delete(bvecs, b0_idx, axis=1)
    #     dotprod = np.abs(bvecs_aux.T @ dti_v1[x, y, z])
    #     idx = np.argsort(dotprod)[::-1]  # np.concatenate([np.array(b0_idx), np.argsort(dotprod)[::-1]]) # Sort them in descending order, keeping the b0 vols first
    #     d_rearranged = data_aux[idx]
    #     data_rearranged[x, y, z] = np.concatenate([data[x, y, z, b0_idx], d_rearranged])

    coords = np.array(np.where(mask != 0))
    if coords.shape[0] == 2:
        coords = np.concatenate((coords, np.ones((1, coords.shape[1]))), axis=0)
    elif coords.shape[0] == 1:
        coords = np.concatenate((coords, np.ones((2, coords.shape[1]))), axis=0)

    data_rearranged = np.zeros_like(data)
    for x,y,z in coords.T:
        b0_idx = [0, 17, 33, 35, 52, 68] # Filter b0 values
        data_aux = np.delete(data[x,y,z], b0_idx)
        bvecs_aux = np.delete(bvecs, b0_idx, axis=1)
        dotprod = np.abs(bvecs_aux.T @ dti_v1[x, y, z])
        idx = np.argsort(dotprod)[::-1]# np.concatenate([np.array(b0_idx), np.argsort(dotprod)[::-1]]) # Sort them in descending order, keeping the b0 vols first
        d_rearranged = data_aux[idx]
        data_rearranged[x,y,z] = np.concatenate([data[x,y,z,b0_idx], d_rearranged])

    # Parallel(n_jobs=njobs, prefer="processes", verbose=6)(
    # delayed(rearrange_voxel)(data, data_rearranged, dti_v1, bvecs, i)
    #     for i in coords.T
    #     )

    import nibabel as nb
    orig_data = nb.load('/Volumes/GoogleDrive/My Drive/Nottingham/Github/CMRR/subj1/nordic_res09_ipat2/data.nii.gz')
    export_nifti(data_rearranged, orig_data, out_path, f'{name}_rearranged.nii.gz')



def rearrange_path(data_file, dti_v1_file, bvecs_file, mask_file, out_path, name='default', njobs=1):
    import numpy as np
    from joblib import Parallel, delayed
    import os
    from tools.prepare_NIFTI import export_nifti, get_data

    data = get_data(data_file)
    dti_v1 = get_data(dti_v1_file)
    bvecs = get_data(bvecs_file)
    mask = get_data(mask_file)

    coords = np.array(np.where(mask != 0))
    if coords.shape[0] == 2:
        coords = np.concatenate((coords, np.ones((1, coords.shape[1]))), axis=0)
    elif coords.shape[0] == 1:
        coords = np.concatenate((coords, np.ones((2, coords.shape[1]))), axis=0)

    data_rearranged = np.zeros_like(data)
    for x,y,z in coords.T:
        b0_idx = [0, 17, 33, 35, 52, 68] # Filter b0 values
        data_aux = np.delete(data[x,y,z], b0_idx)
        bvecs_aux = np.delete(bvecs, b0_idx, axis=1)
        dotprod = np.abs(bvecs_aux.T @ dti_v1[x, y, z])
        idx = np.argsort(dotprod)[::-1]# np.concatenate([np.array(b0_idx), np.argsort(dotprod)[::-1]]) # Sort them in descending order, keeping the b0 vols first
        d_rearranged = data_aux[idx]
        data_rearranged[x,y,z] = np.concatenate([data[x,y,z,b0_idx], d_rearranged])

    # Parallel(n_jobs=njobs, prefer="processes", verbose=6)(
    # delayed(rearrange_voxel)(data, data_rearranged, dti_v1, bvecs, i)
    #     for i in coords.T
    #     )

    import nibabel as nb
    orig_data = nb.load('/Volumes/GoogleDrive/My Drive/Nottingham/Github/CMRR/subj1/nordic_res09_ipat2/data.nii.gz')
    export_nifti(data_rearranged, orig_data, out_path, f'{name}_rearranged.nii.gz')
