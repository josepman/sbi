
def brain_from_posterior(posterior_nsf, dataFolder, out_path=None, nsamples=100, s0_est=False, normalization=True, signal_correct=True, n_jobs=6):
    from tools.prepare_NIFTI import get_data
    import os
    from tools.make_dyads import make_dyads
    from joblib import Parallel, delayed
    import torch
    import numpy as np
    from tools.prepare_NIFTI import export_nifti
    import nibabel as nb
    import shutil
    import datetime
    import time

    #### TO DO:
    # - Pass a list with the b0 volumes indices - Now is hardcoded to our specific dataset
    #     - Adapt accordingly the normalization and signal correction functions defined here
    #out_path = '/home/mzxjm1/Desktop/code_modified/sbi_MRI/results/1M/'
    #posterior_nsf = torch.load('/home/mzxjm1/Desktop/code_modified/sbi_MRI/results/posterior_50k_SNR30_NSF_U(1e-4, 75e-3).npy')
    #data_brain = get_data('/home/mzxjm1/Desktop/code_modified/data/data/data.nii.gz')[:, :, 28]
    #brain_mask = get_data('/home/mzxjm1/Desktop/code_modified/data/data/nodif_brain_mask_ero2.nii.gz')[:, :, 28]
    #FA_mask = get_data('/home/mzxjm1/Desktop/code_modified/data/data/FA_mask_ero.nii.gz')  # [:, :, 28]

    ## INITIALIZE MEMMAPS FILES
    memmap_folder = f'{out_path}/memmap_folder'
    if os.path.isdir(memmap_folder):
        pass
    else:
        os.mkdir(memmap_folder)

    data_brain = get_data(f'{dataFolder}/data/data.nii.gz', mmap=True)
    brain_mask = get_data(f'{dataFolder}/data/nodif_brain_mask_ero.nii.gz', mmap=True)

    mean_d = np.memmap(os.path.join(memmap_folder, 'd_mean_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2]), dtype='float')
    std_d = np.memmap(os.path.join(memmap_folder, 'd_std_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2]), dtype='float')
    samples_d = np.memmap(os.path.join(memmap_folder, 'd_samples_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2], nsamples), dtype='float')
    mean_f1 = np.memmap(os.path.join(memmap_folder, 'f1_mean_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2]), dtype='float')
    std_f1 = np.memmap(os.path.join(memmap_folder, 'f1_std_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2]), dtype='float')
    samples_f1 = np.memmap(os.path.join(memmap_folder, 'f1_samples_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2], nsamples), dtype='float')
    disp1 = np.memmap(os.path.join(memmap_folder, 'disp1_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2]), dtype='float')
    v1 = np.memmap(os.path.join(memmap_folder, 'v1_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2], 3), dtype='float')

    if s0_est:
        mean_s0 = np.memmap(os.path.join(memmap_folder, 's0_mean_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2]), dtype='float')
        std_s0 = np.memmap(os.path.join(memmap_folder, 's0_std_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2]), dtype='float')
        samples_s0 = np.memmap(os.path.join(memmap_folder, 's0_samples_memmap'), mode='w+', shape=(data_brain.shape[0], data_brain.shape[1], data_brain.shape[2], nsamples), dtype='float')


    ## PREPARE DATA
    ax_signal = len(data_brain.shape) - 1
    if normalization:  # by b0 volumes
        b0vols = np.take(data_brain, [0, 1, 2, 3, 4], ax_signal)  # b0_vols = data_brain[:, :, 0:5]
        mean_b0vols = np.mean(b0vols, axis=ax_signal)  
        data_brain = data_brain / np.expand_dims(mean_b0vols, axis=ax_signal)

    if signal_correct:
        def correct_signal(signal):
            signal = signal.ravel()
            outliers = np.argwhere(
                signal[5:] >= max(signal[:5])) + 5  # Exclude the b0 volumes
            if len(outliers) > 0:
                corrected_val = min(signal[:5]) - max(signal[:5]) / 100
                signal[outliers] = corrected_val
            # signal[signal >= max(signal[:5])] = max(signal[:5]) - max(signal[:5]) / 100
            # signal[signal <= 0] = 0.0001
            # signal = (signal - min(signal)) / (max(signal) - min(signal))
            return signal

        data_brain = np.apply_along_axis(correct_signal, ax_signal, data_brain)


    def run_brain(posterior_nsf, data_brain, coords, samples_d, samples_f1, mean_d, mean_f1, std_d, std_f1, v1, disp1, mean_s0=None, std_s0=None, samples_s0=None):
        x, y, z = coords.astype(int)
        signal = data_brain[x, y, z].copy()
        if all(signal[5:]<min(signal[:5])):
            s = posterior_nsf.sample((nsamples,), x=signal, sample_with_mcmc=False)
            mean_d[x,y,z] = torch.mean(s[:, 0])
            std_d[x, y, z] = torch.std(s[:, 0])
            samples_d[x, y, z] = np.array(s[:, 0])

            mean_f1[x,y,z] = torch.mean(s[:, 1])
            std_f1[x,y,z] = torch.std(s[:, 1])
            samples_f1[x, y, z] = np.array(s[:, 1])
            v1[x,y,z], disp1[x,y,z] = make_dyads(np.array(s[:, 2]), np.array(s[:, 3]))

            if s0_est:
                std_s0[x, y, z] = torch.std(s[:, -1])
                mean_s0[x, y, z] = torch.mean(s[:, -1])
                samples_s0[x,y,z] = np.array(s[:,-1])


    coords = np.array(np.where(brain_mask != 0))
    if coords.shape[0] == 2:
        coords = np.concatenate((coords, np.ones((1, coords.shape[1]))), axis=0)
    elif coords.shape[0] == 1:
        coords = np.concatenate((coords, np.ones((2, coords.shape[1]))), axis=0)


    print(f' START AT {datetime.datetime.now()}')
    start_time = time.time()
    Parallel(n_jobs=n_jobs, prefer="processes", verbose=6)(
        delayed(run_brain)(posterior_nsf, data_brain, coords[:, i], samples_d, samples_f1, mean_d, mean_f1, std_d, std_f1, v1, disp1, mean_s0, std_s0, samples_s0)
        for i in range(0, coords.shape[1])
        )
    print(f' FINISHED AT {datetime.datetime.now()}')
    print(f'time elapsed: {time.time() - start_time}')

    data_brain_orig = nb.load(f'{dataFolder}/data/data.nii.gz')
    export_nifti(mean_d, data_brain_orig, out_path, 'mean_dsamples.nii.gz')
    export_nifti(std_d, data_brain_orig, out_path, 'std_d.nii.gz')
    export_nifti(mean_f1, data_brain_orig, out_path, 'mean_f1samples.nii.gz')
    export_nifti(std_f1, data_brain_orig, out_path, 'std_f1.nii.gz')
    export_nifti(v1, data_brain_orig, out_path, 'v1_dyads.nii.gz')
    export_nifti(disp1, data_brain_orig, out_path, 'disp1.nii.gz')
    export_nifti(samples_d, data_brain_orig, out_path, 'merged_dsamples.nii.gz')
    export_nifti(samples_f1, data_brain_orig, out_path, 'merged_f1samples.nii.gz')

    if s0_est:
        export_nifti(mean_s0, data_brain_orig, out_path, 'mean_s0samples.nii.gz')
        export_nifti(std_s0, data_brain_orig, out_path, 'std_s0.nii.gz')
        export_nifti(samples_s0, data_brain_orig, out_path, 'merged_s0samples.nii.gz')

    shutil.rmtree(memmap_folder)
    print('Done!')
