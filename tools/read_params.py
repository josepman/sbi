from argparse import ArgumentParser, SUPPRESS
import os
import sys
import pathlib
from runpy import run_path
from datetime import datetime
import numpy as np
import pandas as pd
import nibabel as nb
from joblib import dump, load
from tools.cart2sph import cart2sph
from tools.prepare_NIFTI import get_data
from tools.reshape_brain import reshape_brain_image, reshape_brain_mask

def extract(dct, namespace=None):
    if not namespace: namespace = globals()
    namespace.update(dct)
    
def unpack_vars(logger, argv):
    # Only CONFIGFILE
    if isinstance(argv, str):   # It is a string, i.e. a path (chain of characters) = Only configfile provided
        if (pathlib.Path(argv).suffix == '.py'):
            args = run_path(argv)
        else:
            logger.info('Reading parameters from configfile....')
            args = read_configfile(argv)
        
    # COMMAND LINE PARAMS
    elif isinstance(argv, list):    # Params introduced by terminal
        logger.info('Reading parameters from command-line....')
        args = read_params(argv)

        
        # # IF THERE IS ALSO A CONFIGFILE, LOAD AND OVERWRITE IT
        # if isinstance(args['configfile'], str):    # Configfile provided also
        #     logger.info('Config file provided! Reading parameters from it....')
        #     logger.info('Note these parameters will overwrite the parameters passed by terminal.')
        #     args_aux = read_configfile(args['configfile'])
        #     # Load all the variables stored inside the dictionary args
        #     extract(args_aux)
        #     # locals().update(args_aux)
        #     # #### However, there are some conflictive variables that we need to redefine manually:
        #     # conflict_vars = ['base_path', 'data_path', 'results_path', 'model_name', 'image_file',
        #     # 'bvals_file', 'bvecs_file', 'brainmask_file', 'header_init', 'reparametrization', 'cons_fit', 'num_sticks']
        #     #
        #     # for i in conflict_vars:
        #     #     try:
        #     #         exec('{} = {}'.format(i, args_aux[i]))
        #     #         print(f'{i} IN ARGS_AUX')
        #     #     except KeyError:
        #     #         print(f'{i} should be passed as param')
        #     #         continue

    
    locals().update(args)
    # print('Variables in locals():')
    # print(locals())
    # for i in list(args.keys()):
    #     exec(f'{i} = {args[i]}')
    # print(locals())
    # locals().update()
    # print(locals())
    #### However, there are some conflictive variables that we need to redefine manually:
    model_name = args['model_name']
    base_path = args['base_path']
    data_path = args['data_path']
    results_path = args['results_path']
    image_file = args['image_file']
    bvals_file = args['bvals_file']
    bvecs_file = args['bvecs_file']
    brainmask_file = args['brainmask_file']
    header_init = args['header_init']
    reparametrization = args['reparametrization']
    cons_fit = args['cons_fit']
    num_sticks = args['num_sticks']
    njobs = args['njobs']
    init_files = args['init_files']
    cov_file = args['cov_file']
    snr = args['snr']
    type_noise = args['type_noise']
    slice_axis = args['slice_axis']
    slice_n = args['slice_n']
    seed = args['seed']
    framework = args['framework']
    flag_fit = args['flag_fit']
    optimizer = args['optimizer']
    ard_optimization = args['ard_optimization']
    likelihood_type = args['likelihood_type']
    run_mcmc = args['run_mcmc']
    mcmc_analysis = args['mcmc_analysis']
    burnin = args['burnin']
    jumps = args['jumps']
    thinning = args['thinning']
    adaptive = args['adaptive']
    period = args['period']
    duration = args['duration']
    ard_mcmc = args['ard_mcmc']
    get_cov_samples = args['get_cov_samples']
    full_report = args['full_report']
    type_ard = args['type_ard']
    ard_fudge = args['ard_fudge']
    bingham_flag = args['bingham_flag']
    acg = args['acg']
    
    t_start = pd.Timestamp.now()  # perf_counter() //time.time()
    model_name = [lambda:model_name, lambda:f'model_{datetime.now().strftime("%H-%M-%S")}'][model_name is None]()    #[lambda: value_false, lambda: value_true][<test>]()
    print(data_path)
    # 1. Data Path --  It should contain the data.nii.gz, bvals, bvecs, and brainmask.nii.gz
    if data_path is None:
        logger.info(f'Data path not selected. Please, check this carefuly!')
        if (os.path.isdir(os.path.join(base_path, 'data/'))):
            data_path = os.path.join(base_path, 'data/')
            sys.path += [data_path]
            logger.info(f'Data path set in from {data_path}')
        else:
            logger.error(f'Error! Data path {data_path} does not exist.')
            sys.exit()

    # 2. Code paths
    gen_model_path = os.path.join(base_path, 'models/')
    mcmc_path = os.path.join(base_path, 'mcmc/')
    vis_path = os.path.join(base_path, 'visualization/')
    utils_path = os.path.join(base_path, 'tools/')
    for i in [gen_model_path, mcmc_path, vis_path, utils_path]:
        if (os.path.isdir(i) == False):
            logger.error(f'Error! {i} does not exist.')
            sys.exit()
        else:
            sys.path += [i]
            logger.info(f'Loading code from {i}')
            

    ## 3. Output paths
    logger.info('Generating output paths')
    results_path = [results_path, os.path.join(base_path, model_name, 'results/')][results_path is None]
    init_results = os.path.join(results_path, 'initialization/')
    dti_folder = os.path.join(init_results, 'DTIFIT/')
    nonlin_fit_folder = os.path.join(init_results, 'PVMFIT/')
    mcmc_results = os.path.join(results_path, 'mcmc/')
    temp_path = os.path.join(results_path, 'temp/')
    memmap_folder = os.path.join(temp_path, '_joblib_memmap/')  # './joblib_memmap'
    for i in [results_path, init_results, dti_folder, nonlin_fit_folder, mcmc_results, temp_path, memmap_folder]:
        pathlib.Path(i).mkdir(parents=True, exist_ok=True)  # Add the folder to the path + Check if the folder exists and if not, create it (and also its parents folder)

    os.rename(f'{os.getcwd()}/main_log.log', f'{results_path}{model_name}.log')  # relocate the log file to the model folder
    sys_path = sys.path

    # Load files
    if image_file is None: image_file = 'data.nii.gz'
    if brainmask_file is None: brainmask_file = 'nodif_brain_mask.nii.gz'
    if bvals_file is None: bvals_file = 'bvals'
    if bvecs_file is None: bvecs_file = 'bvecs'
    aux = nb.load(f'{data_path}{image_file}')
    image = aux.get_data()  #get_data(f'{data_path}{image_file}')
    affine_matrix = aux.affine  # We extract the header of the original image to create new ones
    brain_mask = nb.load(f'{data_path}{brainmask_file}').get_data()
    bvals = np.loadtxt(f'{data_path}{bvals_file}')
    bvecs = np.loadtxt(f'{data_path}{bvecs_file}')

    # Init params
    if header_init is None: header_init = 'pvmfit'
    n_params = 2 + 3 * num_sticks
    max_float = sys.float_info.max
    min_float = sys.float_info.min
    if reparametrization:
        lb = min_float * np.ones((n_params))
        ub = max_float * np.ones((n_params))
    if cons_fit:
        lb = np.array([min_float, min_float] + num_sticks* [-max_float, -max_float, min_float] )
        ub = np.array([max_float, 1] + num_sticks * [max_float, max_float, 1])
    param_bounds = np.array([lb, ub])

    # Reshape inputs if needed to make it compatible with this version of code (it works with 4D images)
    image = reshape_brain_image(image)
    brain_mask = reshape_brain_mask(brain_mask)
    
    #########################################################################################################
    #########################################################################################################
    ## Dump the image for a faster processing when sharing data in the multithread processing
    image_filename_memmap = os.path.join(memmap_folder, 'image_memmap')
    brain_mask_filename_memmap = os.path.join(memmap_folder, 'brain_mask_memmap')
    dump(image, image_filename_memmap)
    dump(brain_mask, brain_mask_filename_memmap)
    brain_mask = load(brain_mask_filename_memmap)
    image = load(image_filename_memmap)
    shape_image = image.shape
    type_image = 'float'  # type(image)
    #########################################################################################################
    #########################################################################################################
    
    
    # Normalize bvecs by gradients
    norm_bvec = np.sqrt(bvecs[0, :] * bvecs[0, :] + bvecs[1, :] * bvecs[1, :] + bvecs[2, :] * bvecs[2, :])
    bvecs = np.array([ bvecs[:,i]/norm_bvec[i] if norm_bvec[i]>0.01 else bvecs[:,i] for i in range(0,len(norm_bvec)) ]).T
    # Convert them to spherical coordinates
    bvecs_sph = np.zeros((2, len(bvals)))
    for i in range(0, len(bvals)):
        _, bvecs_sph[0, i], bvecs_sph[1, i] = cart2sph(bvecs[0, i], bvecs[1, i], bvecs[2, i])

    logger.info(f'Environment initialized in ({pd.Timestamp.now() - t_start})')
    logger.info(f'Errors found: ')
    logger.info(f'Warnings found: ')

    # Unpack variables. By row:
    # paths - files - data - params - other exec params - fit params - mcmc params (I and II)
    return base_path, data_path, results_path, init_results, dti_folder, nonlin_fit_folder, mcmc_results, temp_path, memmap_folder, \
           model_name, image_file, bvals_file, bvecs_file, brainmask_file, header_init, init_files, cov_file, \
           image, affine_matrix, bvals, bvecs, bvecs_sph, brain_mask, image_filename_memmap, brain_mask_filename_memmap, \
           num_sticks, snr, type_noise, param_bounds, n_params, shape_image, type_image, \
           slice_axis, slice_n, seed, njobs, full_report, logger, sys_path, \
           framework, flag_fit, optimizer, cons_fit, ard_optimization, likelihood_type, reparametrization, \
           run_mcmc, mcmc_analysis, burnin, jumps, thinning, adaptive, period, duration, ard_mcmc, \
           get_cov_samples, type_ard, ard_fudge, bingham_flag, acg
           


def read_configfile(configfile):
    # Traditional method using configparser needs to redefine all the variable types, as they are imported as strings
    # import configparser
    # config = configparser.RawConfigParser()
    # config.read(configfile)
    # namevars = list(config['params'].keys())  # Name of params
    # values = [config['params'][x] for x in namevars]  # Put all the params in a list
    # args = zip(namevars, values)
    # Here we should reformat all the variables or have used before .getint, .getfloat, .getbool when reading params

    # localconfig is a wrapper on top of configparser (so fully compatible) that makes easier to import the variables in correct data types using the same configfile
    # https://pypi.org/project/localconfig/
    from localconfig import config
    config.read(configfile)
    args = dict(list(config.args))    # returns a dict
    return args



def read_params(args=None):
    parser = ArgumentParser(prog='hy_bedpostX.py', description='Routine for Fibres orientation estimation from pre-processed dMRI data',
                                     add_help=False, allow_abbrev=True, argument_default=None, prefix_chars='-',
                                     epilog='For more information, see XXXXX.com')
    
    # Small hack to show required options even using the -- notation:
    # Extracted from https://stackoverflow.com/questions/24180527/argparse-required-arguments-listed-under-optional-arguments
    required = parser.add_argument_group('REQUIRED ARGUMENTS')
    optional = parser.add_argument_group('OPTIONAL ARGUMENTS')
    # Add back help
    optional.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit')
    
    # Paths, names and files
    required.add_argument('-i', '--data_path', help='Data folder containing data.nii.gz, bvals, bvecs and brainmask.nii.gz', required=True)
    required.add_argument('-o', '--results_path', help='Results folder. If it does not exist, it is created', required=True)
    optional.add_argument('--base_path', default=os.getcwd(), help='Folder containing the code')       ### Cambiar folder por la carpeta del codigo
    optional.add_argument('-cf', '--configfile', dest='configfile', help="Configuration file with params definition inside")
    optional.add_argument('--image_file', default='data.nii.gz', help='Data filename. Default data.nii.gz')
    optional.add_argument('--bvals_file', default='bvals', help='Data filename. Default bvals')
    optional.add_argument('--bvecs_file', default='bvecs', help='Data filename. Default bvecs')
    optional.add_argument('--brainmask_file', default='nodif_brain_mask.nii.gz', help='Brainmask filename. Default nodif_brain_mask.nii.gz')
    optional.add_argument('--model_name', default=None, help='Name given to the experiment. Default model_<time>')
    optional.add_argument('--header_init', default='pvmfit', help='Header used for the fitted params files. Default pvmfit')
    optional.add_argument('--init_files', default=None, help='Path to the folder with the non_linear fitted results (e.g. PVMFIT folder)')
    optional.add_argument('--cov_file', default='Analytic', help='Covariance used for Hybrid, Indep. Sampler and Block Approaches. It can be provided or calculated. Default Analytic')
    
    # Other execution parameters
    optional.add_argument('--njobs', default=1, dest='njobs', help='Number of cores to use (parallelization). Default 1')
    optional.add_argument('--slice_axis', default=None, choices=['x', 'y', 'z', 'None'], metavar = 'z', help='To run only 1 slice in {x,y,z} axis. Default None')
    optional.add_argument('--slice_n', default=None, type=int, metavar = '28', help='Slice n in axis [--slice_axis]. Default None')
    optional.add_argument('--seed', default=1234, type=int, help='For pseudo random number generator. Default 1234')
    optional.add_argument('--get_cov_samples', action="store_true", help='FLAG: To extract samples covariance in each voxel')
    optional.add_argument('--full_report', action="store_true", help='FLAG: To export all the outputs (jacobian, reports, etc.)')
    
    # General model parameters
    optional.add_argument('-n', '--num_sticks', type=int, default=3, metavar = '3', help='Number of fibres per voxel, default 3')
    optional.add_argument('--snr', metavar = '30', default=30)
    optional.add_argument('--type_noise', choices=['Gaussian', 'Rician'], default='Rician', metavar = 'Rician',  help='Noise model. Default Rician')
    
    # Fitting parameters
    optional.add_argument('--framework', choices=['Python', 'Matlab', 'FSL'], default='Matlab', metavar = 'Matlab', help='Where to run the DTIFIT and PVMFIT. Default Matlab')
    optional.add_argument('--flag_fit', choices=['all', 'PVMFIT', 'DTIFIT'], default='all', metavar = 'flag_fit', help='Only for FSL framework. Whether to run DTIFIT, PVMFIT or both.')
    optional.add_argument('--optimizer', choices=['fmincon'], default='fmincon', metavar = 'fmincon', help='Optimizer used in the non-linear fitting. See doc for more info.')
    optional.add_argument('--cons_fit', action="store_true", help='FLAG: Perform constrained fitting (see doc for changing boundaries).')
    optional.add_argument('--ard_optimization', action="store_true", help='FLAG: To include the ARD prior in the non-linear fitting (only for Matlab and Python frameworks)')
    optional.add_argument('--likelihood_type', default='Default')
    optional.add_argument('--reparametrization', action="store_true", help='FLAG: To do reparametrization. See doc.')
    
    # MCMC parameters
    optional.add_argument('--no_run_mcmc', dest='run_mcmc', action="store_false", help='FLAG: To not run the MCMC routine')
    optional.add_argument('--mcmc_analysis', default='Hybrid1', choices=['Component', 'Hybrid1', 'Hybrid2', 'Block', 'Laplace'], metavar = 'Hybrid1', help='Type of MCMC. Default Hybrid1. See doc for more information')
    optional.add_argument('-b', '--burnin', type=int, default=1000, metavar = '1000', help='Burnin period, default 1000')
    optional.add_argument('-j', '--jumps', type=int, default=1250, metavar = '1250', help='Number of jumps after burnin, default 1250')
    optional.add_argument('-s', '--thinning', default=25, type=int, metavar = '25', help='Sample every (i.e., thinning), default 25')
    optional.add_argument('-a', '--adaptive', action="store_true", help='FLAG: To activate the adaptive MCMC.')
    optional.add_argument('-L', '--period', type=int, default=50, metavar = '50', help='Adapt proposal density (std or cov) every L jumps, default 50. "adaptive" must be activated.')
    optional.add_argument('--duration', choices=['burnin', 'whole-run'], metavar = 'whole-run', help='Perform adaptive MCMC during burnin period (recommended) or during the whole run (FSL). "adaptive" must be activated.', default='whole-run')
    optional.add_argument('--no_ard_mcmc', dest='ard_mcmc', action="store_false", help='FLAG: Turn ARD off on all fibres in the MCMC')
    optional.add_argument('--type_ard', choices=['Beta', 'Gaussian'], default='Gaussian', metavar = 'Gaussian', help='Distribution used to derive the ARD prior. Default Gaussian') #old ard_flag
    optional.add_argument('--ard_fudge', metavar = '1', default=1)
    optional.add_argument('--bingham_flag', action="store_true", help="FLAG")
    optional.add_argument('--acg', action="store_true", help="FLAG")
    
    ## ensure positivity in some parameters
    
    return vars(parser.parse_args(args))    # it returns a dict


