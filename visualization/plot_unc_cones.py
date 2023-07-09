def plot_unc_cones(brain_mask, th_samples, ph_samples, CI=95, th2_samples=None, ph2_samples=None, aff_mat=None, name1='v1', name2='v2', savefig=None):
    '''It calculates the angle of the uncertainty cones of the orientations provided.
    Orientations needs to be in spherical coordinates. Use the function tools.cart2sph to transform them if needed.
    If savefig is provided, .png files with the images are exported (in savefig path) but not shown.
    If a second orientation map is provided, it plots also the difference between them.
    If affine matrix is provided, the error maps are also exported in nifti format'''
    
    import numpy as np
    import matplotlib.pyplot as plt
    import nibabel as nb
    from tools.get_unc_cones import get_unc_cones_map
    from visualization.plot_density_scatter import density_scatter
    
    CI_map = np.nan_to_num(get_unc_cones_map(th_samples, ph_samples, brain_mask, CI=CI))
    fig = plt.figure()
    plt.imshow(CI_map, cmap='bwr'), plt.title(f'{name1}'), plt.colorbar(), plt.show()
    if savefig:
        fig.savefig(f'{savefig}unc_map_{name1}_{CI}.png')
        if aff_mat is not None:
            nb.save(nb.Nifti2Image(CI_map, affine=aff_mat), f'{savefig}unc_map_{name1}_{CI}.nii.gz')
    else:
        plt.show()
        
    # If another vector of orientations is provided, we can compare them
    if ((th2_samples is not None) and (ph2_samples is not None)):
        CI_map2 = np.nan_to_num(get_unc_cones_map(th_samples, ph_samples, brain_mask, CI=CI))
        fig = plt.figure()
        plt.imshow(CI_map2, cmap='bwr'), plt.title(f'{name2}'), plt.colorbar(), plt.show()
        if savefig:
            fig.savefig(f'{savefig}unc_map_{name2}_{CI}.png')
            if aff_mat is not None:
                nb.save(nb.Nifti2Image(CI_map2, affine=aff_mat), f'{savefig}unc_map_{name2}_{CI}.nii.gz')
        else:
            plt.show()
        
        difference = CI_map - CI_map2
        fig = plt.figure()
        plt.imshow(difference, cmap='bwr'), plt.title(f'{name1}-{name2}'), plt.colorbar(), plt.show()
        if savefig:
            fig.savefig(f'{savefig}unc_map_difference_{CI}.png')
            if aff_mat is not None:
                nb.save(nb.Nifti2Image(difference, affine=aff_mat), f'{savefig}unc_map_difference_{CI}.nii.gz')
        else:
            plt.show()
            
        fig = plt.figure()
        density_scatter(CI_map[brain_mask > 0], CI_map2[brain_mask > 0], cmap='bwr'), plt.title(f'{name1} vs. {name2}'), plt.show()
        if savefig:
            fig.savefig(f'{savefig}scatterplot_{CI}.png')
        else:
            plt.show()

    