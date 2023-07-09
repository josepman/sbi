import numpy as np
from pathlib import Path
import os
import torch
from sbi.inference import prepare_for_sbi, SNPE_C
from joblib import Parallel, delayed
from src.tools.prepare_NIFTI import get_data
from src.tools.get_priors import get_priors
from src.brain_from_posterior import brain_from_posterior
from datetime import datetime
from sbi.inference import simulate_for_sbi
from src.models.ballSticks import simulator_class, simulator_normalized, simulator_nos0

def main(training_params, brain_inference_params):
    ## INITIALIZATION
    gamma, s0_est, alpha, beta, d_min, d_max, normalization, signal_correct, n_sim, num_fib, njobs, resultsFolder = training_params

    ## PRIORS
    priors_simulation, priors_inference, name = get_priors(gamma, s0_est, alpha, beta, d_min, d_max, num_fib, n_sim)

    if normalization:
        name += '_norm'
    if signal_correct:
        name += '_corr'

    if os.path.isdir(f'{resultsFolder}/{name}'):
        pass
    else:
        os.mkdir(f'{resultsFolder}/{name}')

    ## PARAMETERS SAMPLING AND SIMULATIONS
    print('Simulating data...')
    print(f'{datetime.now().strftime("%D %H:%M:%S")}')
    simulator = simulator_class()
    if normalization:
        simulator = simulator_normalized()
    if s0_est==False:
        simulator = simulator_nos0()
    simulator, prior = prepare_for_sbi(simulator, priors_inference)

    theta, x_presim = simulate_for_sbi(
            simulator=simulator,
            proposal=priors_simulation,
            num_simulations=n_sim,
            num_workers=njobs,
            show_progress_bar=True
        )
    
    # Correct angular params to make uniform samples on the sphere
    for i in range(0, num_fib):
            theta[:, 2 + 3 * i] = np.arccos(1 - 2 * theta[:, 2 + 3 * i])  # v from cart to sph
            theta[:, 3 + 3 * i] = 2 * np.pi * theta[:, 3 + 3 * i]

    ## TRAINING
    run_in_device = 'cpu'
    print('Starting Inference:')
    print(f'{datetime.now().strftime("%D %H:%M:%S")}')
    inference = SNPE_C(priors_inference, density_estimator='mdn', device=run_in_device,
                           logging_level='INFO', show_progress_bars=True)  # , summary_writer=True)
    _ = inference.append_simulations(theta, x_presim).train(max_num_epochs=10)  
    posterior_nsf = inference.build_posterior() # Default: density_estimator=None, rejection_sampling_parameters=None, sample_with_mcmc=False, mcmc_method='slice_np', mcmc_parameters=None)


    print(f'Inference finished at: {datetime.now().strftime("%D %H:%M:%S")}')
    # torch.save(posterior_nsf, f'{results_path}/{name}/posterior_{name}.npy')

    ## INFERENCE FROM BRAIN DATA
    inference_brain_data, dataFolder, nsamples_brain, njobs_brain = brain_inference_params
    if inference_brain_data:
        brain_from_posterior(posterior_nsf, dataFolder, f'{resultsFolder}/{name}', nsamples_brain, s0_est, normalization, signal_correct, njobs_brain)


if __name__ == '__main__':
    # Training params
    gamma = False   # If false, prior_d = uniform
    if gamma:
        alpha = 3.5     # Params of gamma distribution
        beta = 1500     # (up to 4000.0)
    else:
        d_min = 0.0001  # Limits of uniform distribution for diffusivity
        d_max = 0.0075  # 
    
    num_fib = 1
    n_sim = 1000    
    normalization = True    # If true, normalize signal by s0
    signal_correct = True   # If true, correct signal clipping negative values
    s0_est = False          # If true, estimate s0
    njobs = 1
    
    basePath = Path(os.getcwd()).parent.absolute()
    dataFolder = f'{basePath}/Data'
    resultsFolder = f'{basePath}/results/{num_fib}fib'

    # Brain inference params
    inference_brain_data = False 
    brain_inference_params = [inference_brain_data]
    if inference_brain_data:
        nsamples_brain = 50     # Number of samples to generate from posterior
        njobs_brain = 1         # Number of jobs to run in parallel
        brain_inference_params.append(dataFolder, nsamples_brain, njobs_brain)

    # Run
    if isinstance(gamma, list): # If want to run multiple configurations (e.g. different priors)
        Parallel(n_jobs=len(gamma), prefer="processes", verbose=6)(
            delayed(main)([gamma[i], s0_est, alpha[i], beta[i], d_min[i], d_max[i], normalization, signal_correct, n_sim, num_fib, njobs, resultsFolder], brain_inference_params)
            for i in range(0, len(gamma))
        )
    else:
        training_params = [gamma, s0_est, alpha, beta, d_min, d_max, normalization, signal_correct, n_sim, num_fib, njobs, resultsFolder]
        main(training_params, brain_inference_params)