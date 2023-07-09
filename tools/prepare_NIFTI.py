def get_data(file):
    """
    Args:
        file:
    """
    import nibabel as nb
    img = nb.load(file)
    img_voxels = img.get_fdata()
    return img_voxels


def export_nifti(data, orig_data, output_path, name):
    """
    Args:
        data:
        orig_data:
        output_path:
        name:
    """
    import nibabel as nb
    import os
    # Copy the header of the original image
    aff_mat = orig_data.affine
    nb.save(nb.Nifti2Image(data, affine=aff_mat), os.path.join(output_path, name))


def export_DTIFIT_results(dataobs_file, output_path, s0_est, MD, MO, FA, L1, L2, L3, V1, V2, V3):
    """
    Args:
        dataobs_file:
        output_path:
        s0_est:
        MD:
        MO:
        FA:
        L1:
        L2:
        L3:
        V1:
        V2:
        V3:
    """
    import nibabel as nb
    from tools.prepare_NIFTI import export_nifti

    orig_data = nb.load(dataobs_file)
    export_nifti(s0_est, orig_data, output_path, 'dtifit_S0.nii.gz')
    export_nifti(MD, orig_data, output_path, 'dtifit_MD.nii.gz')
    export_nifti(MO, orig_data, output_path, 'dtifit_MO.nii.gz')
    export_nifti(FA, orig_data, output_path, 'dtifit_FA.nii.gz')
    export_nifti(L1, orig_data, output_path, 'dtifit_L1.nii.gz')
    export_nifti(L2, orig_data, output_path, 'dtifit_L2.nii.gz')
    export_nifti(L3, orig_data, output_path, 'dtifit_L3.nii.gz')
    export_nifti(V1, orig_data, output_path, 'dtifit_V1.nii.gz')
    export_nifti(V2, orig_data, output_path, 'dtifit_V2.nii.gz')
    export_nifti(V3, orig_data, output_path, 'dtifit_V3.nii.gz')


def export_PVMFIT_results(dataobs_file, output_path, num_sticks, prefix_name, initial_values):
    """
    Args:
        dataobs_file:
        output_path:
        num_sticks:
        prefix_name:
        initial_values:
    """
    import numpy as np
    import nibabel as nb
    from tools.prepare_NIFTI import export_nifti  # , PVMFIT_comparisons
    from tools.sph2cart import sph2cart
    orig_data = nb.load(dataobs_file)

    export_nifti(initial_values[:, :, :, 0], orig_data, output_path, prefix_name + '_S0.nii.gz')
    export_nifti(initial_values[:, :, :, 1], orig_data, output_path, prefix_name + '_d.nii.gz')
    export_nifti(initial_values[:, :, :, 2], orig_data, output_path, prefix_name + '_th1.nii.gz')
    export_nifti(initial_values[:, :, :, 3], orig_data, output_path, prefix_name + '_ph1.nii.gz')
    export_nifti(initial_values[:, :, :, 4], orig_data, output_path, prefix_name + '_f1.nii.gz')
    v1 = np.zeros((initial_values[:, :, :, 2].shape[0], initial_values[:, :, :, 2].shape[1],
                   initial_values[:, :, :, 2].shape[2], 3))
    v1[:, :, :, 0], v1[:, :, :, 1], v1[:, :, :, 2] = sph2cart(initial_values[:, :, :, 2], initial_values[:, :, :, 3])
    export_nifti(v1, orig_data, output_path, prefix_name + '_V1.nii.gz')
    if num_sticks == 3:
        export_nifti(initial_values[:, :, :, 5], orig_data, output_path, prefix_name + '_th2.nii.gz')
        export_nifti(initial_values[:, :, :, 6], orig_data, output_path, prefix_name + '_ph2.nii.gz')
        export_nifti(initial_values[:, :, :, 7], orig_data, output_path, prefix_name + '_f2.nii.gz')
        v2 = np.zeros((initial_values[:, :, :, 5].shape[0], initial_values[:, :, :, 5].shape[1],
                       initial_values[:, :, :, 5].shape[2], 3))
        v2[:, :, :, 0], v2[:, :, :, 1], v2[:, :, :, 2] = sph2cart(initial_values[:, :, :, 5],
                                                                  initial_values[:, :, :, 6])
        export_nifti(v2, orig_data, output_path, prefix_name + '_V2.nii.gz')

        export_nifti(initial_values[:, :, :, 8], orig_data, output_path, prefix_name + '_th3.nii.gz')
        export_nifti(initial_values[:, :, :, 9], orig_data, output_path, prefix_name + '_ph3.nii.gz')
        export_nifti(initial_values[:, :, :, 10], orig_data, output_path, prefix_name + '_f3.nii.gz')
        v3 = np.zeros((initial_values[:, :, :, 8].shape[0], initial_values[:, :, :, 8].shape[1],
                       initial_values[:, :, :, 8].shape[2], 3))
        v3[:, :, :, 0], v3[:, :, :, 1], v3[:, :, :, 2] = sph2cart(initial_values[:, :, :, 8],
                                                                  initial_values[:, :, :, 9])
        export_nifti(v3, orig_data, output_path, prefix_name + '_V3.nii.gz')

    elif num_sticks == 2:
        export_nifti(initial_values[:, :, :, 5], orig_data, output_path, prefix_name + '_th2.nii.gz')
        export_nifti(initial_values[:, :, :, 6], orig_data, output_path, prefix_name + '_ph2.nii.gz')
        export_nifti(initial_values[:, :, :, 7], orig_data, output_path, prefix_name + '_f2.nii.gz')
        v2 = np.zeros((initial_values[:, :, :, 5].shape[0], initial_values[:, :, :, 5].shape[1],
                       initial_values[:, :, :, 5].shape[2], 3))
        v2[:, :, :, 0], v2[:, :, :, 1], v2[:, :, :, 2] = sph2cart(initial_values[:, :, :, 5],
                                                                  initial_values[:, :, :, 6])
        export_nifti(v2, orig_data, output_path, prefix_name + '_V2.nii.gz')


def mcmc_to_nifti(aff_mat, output_path, mcmc_objects, maps):
    """
    Args:
        aff_mat:
        output_path:
        mcmc_objects:
        maps:
    """
    import nibabel as nb
    print('Exporting f1 (mean and std)')
    nb.save(nb.Nifti2Image(mcmc_objects[1], affine=aff_mat), output_path + 'mean_s0samples.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[2], affine=aff_mat), output_path + 'std_s0samples.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[4], affine=aff_mat), output_path + 'mean_dsamples.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[5], affine=aff_mat), output_path + 'std_dsamples.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[8], affine=aff_mat), output_path + 'dyads1.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[9], affine=aff_mat), output_path + 'dyads1_dispersion.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[11], affine=aff_mat), output_path + 'mean_f1samples.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[12], affine=aff_mat), output_path + 'std_f1samples.nii.gz')
    print('Exporting f1 (samples)')
    nb.save(nb.Nifti2Image(mcmc_objects[0], affine=aff_mat), output_path + 'merged_s0samples.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[3], affine=aff_mat), output_path + 'merged_dsamples.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[6], affine=aff_mat), output_path + 'merged_th1samples.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[7], affine=aff_mat), output_path + 'merged_phi1samples.nii.gz')
    nb.save(nb.Nifti2Image(mcmc_objects[10], affine=aff_mat), output_path + 'merged_f1samples.nii.gz')

    if len(mcmc_objects) == 27:
        print('Exporting f2 (mean and std)')
        nb.save(nb.Nifti2Image(mcmc_objects[15], affine=aff_mat), output_path + 'dyads2.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[16], affine=aff_mat), output_path + 'dyads2_dispersion.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[18], affine=aff_mat), output_path + 'mean_f2samples.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[19], affine=aff_mat), output_path + 'std_f2samples.nii.gz')
        print('Exporting f2 (samples)')
        nb.save(nb.Nifti2Image(mcmc_objects[13], affine=aff_mat), output_path + 'merged_th2samples.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[14], affine=aff_mat), output_path + 'merged_phi2samples.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[17], affine=aff_mat), output_path + 'merged_f2samples.nii.gz')
        print('Exporting f3 (mean and std)')
        nb.save(nb.Nifti2Image(mcmc_objects[22], affine=aff_mat), output_path + 'dyads3.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[23], affine=aff_mat), output_path + 'dyads3_dispersion.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[25], affine=aff_mat), output_path + 'mean_f3samples.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[26], affine=aff_mat), output_path + 'std_f3samples.nii.gz')
        print('Exporting f3 (samples)')
        nb.save(nb.Nifti2Image(mcmc_objects[24], affine=aff_mat), output_path + 'merged_f3samples.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[20], affine=aff_mat), output_path + 'merged_th3samples.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[21], affine=aff_mat), output_path + 'merged_phi3samples.nii.gz')

        mean_fsum = mcmc_objects[11] + mcmc_objects[18] + mcmc_objects[25]
        nb.save(nb.Nifti2Image(mean_fsum, affine=aff_mat), output_path + 'mean_fsum.nii.gz')
    elif len(mcmc_objects) == 20:
        print('Exporting f2 (mean and std)')
        nb.save(nb.Nifti2Image(mcmc_objects[15], affine=aff_mat), output_path + 'dyads2.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[16], affine=aff_mat), output_path + 'dyads2_dispersion.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[18], affine=aff_mat), output_path + 'mean_f2samples.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[19], affine=aff_mat), output_path + 'std_f2samples.nii.gz')
        print('Exporting f2 (samples)')
        nb.save(nb.Nifti2Image(mcmc_objects[13], affine=aff_mat), output_path + 'merged_th2samples.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[14], affine=aff_mat), output_path + 'merged_phi2samples.nii.gz')
        nb.save(nb.Nifti2Image(mcmc_objects[17], affine=aff_mat), output_path + 'merged_f2samples.nii.gz')

        mean_fsum = mcmc_objects[11] + mcmc_objects[18]
        nb.save(nb.Nifti2Image(mean_fsum, affine=aff_mat), output_path + 'mean_fsum.nii.gz')

    if maps[0]:
        full_report, acc_ratio_map, bound_rej_map, MH_rej_map, hist_samples, hist_decision, likelihood_hist = maps
        if hist_samples is not None:
            import numpy as np
            nb.save(nb.Nifti2Image(hist_samples, affine=aff_mat), output_path + 'hist_samples.nii.gz')
        if hist_decision is not None:
            import numpy as np
            nb.save(nb.Nifti2Image(hist_samples, affine=aff_mat), output_path + 'hist_decision.nii.gz')
        if likelihood_hist is not None:
            import numpy as np
            nb.save(nb.Nifti2Image(hist_samples, affine=aff_mat), output_path + 'likelihood_hist.nii.gz')
    else:
        full_report, acc_ratio_map, bound_rej_map, MH_rej_map = maps

    if acc_ratio_map is not None:
        nb.save(nb.Nifti2Image(acc_ratio_map, affine=aff_mat), output_path + 'acc_ratio_map.nii.gz')
    if bound_rej_map is not None:
        nb.save(nb.Nifti2Image(bound_rej_map, affine=aff_mat), output_path + 'bound_rej_map.nii.gz')
    if MH_rej_map is not None:
        nb.save(nb.Nifti2Image(MH_rej_map, affine=aff_mat), output_path + 'MH_rej_map.nii.gz')
