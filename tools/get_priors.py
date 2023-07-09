def get_priors(noise_est, s0_est, gamma, num_fib):
    import numpy as np
    import torch
    import sbi.utils
    from sbi.utils.user_input_checks_utils import MultipleIndependent
    torch.manual_seed(0)

    name = ''
    prior_min_sim = []
    prior_max_sim = []
    priors_simulation = []
    priors_inference = []

    # If signal is not normalized and s0 is estimated 
    if s0_est:
        prior_min_sim.append(1.0)
        prior_max_sim.append(15000.0)
        
    # f, th, ph
    prior_min_sim.extend(num_fib * [0.0, 0.0, 0.0])  
    prior_min_inf = prior_min_sim.copy()
    prior_max_inf = prior_max_sim.copy()
    prior_max_sim.extend(num_fib * [1.0, 1.0, 1.0])
    prior_max_inf.extend(num_fib * [1.0, np.pi, 2 * np.pi])

    if noise_est:
        prior_min_sim.append(5.0)
        prior_max_sim.append(55.0)
        prior_min_inf.append(5.0)
        prior_max_inf.append(55.0)
    
    # Gamma prior used for d
    if gamma:
        #from sbi.user_input.user_input_checks_utils import MultipleIndependent
        alpha = 3.5 #[0, 3.5, 3.0, 3.5, 4.0, 5.0, 0,0,0,0]
        beta = 1500 #[0, 1500.0, 4000.0, 1500.0, 2000.0, 1500.0, 0,0,0,0]
        name = f'_G({alpha},{int(beta)})'
        priors_simulation.append(torch.distributions.gamma.Gamma(torch.tensor([alpha]), torch.Tensor([beta])))
        [priors_simulation.append(sbi.utils.torchutils.BoxUniform(low=torch.Tensor([prior_min_sim[i]]), high=torch.Tensor([prior_max_sim[i]]))) for i in range(0, len(prior_max_sim))]
        priors_simulation = MultipleIndependent(priors_simulation)

        priors_inference.append(torch.distributions.gamma.Gamma(torch.tensor([alpha]), torch.Tensor([beta])))
        [priors_inference.append(sbi.utils.torchutils.BoxUniform(low=torch.Tensor([prior_min_inf[i]]), high=torch.Tensor([prior_max_inf[i]]))) for i in range(0, len(prior_max_inf))]
        priors_inference = MultipleIndependent(priors_inference)

    # Uniform prior used for d
    else:
        d_min = 0.0001 #[0.0001, 0, 0,0,0,0, 0.0, 0.0, 0.0001, 0.0001]
        d_max = 0.0075 #[0.0075, 0, 0,0,0,0, 0.005, 0.0075, 0.005, 0.0075]
        name = f'_U({d_min},{d_max})'
        prior_min_sim.insert(0, d_min) #[d_min] + num_fib*[0.0, 0.0, 0.0]  # d, f, th, ph
        prior_max_sim.insert(0, d_max) #[d_max] + num_fib*[1.0, 1.0, 1.0]
        prior_min_inf.insert(0, d_min) 
        prior_max_inf.insert(0, d_max)
        
        priors_simulation = sbi.utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min_sim), high=torch.as_tensor(prior_max_sim))
        priors_inference = sbi.utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min_inf), high=torch.as_tensor(prior_max_inf))
    
    if noise_est:
        name+= '_noise'
    # params returned: d, s0, f_i, th_i, ph_i, .., SNR
    return priors_simulation, priors_inference, name