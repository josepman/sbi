class simulator_class:

    def __init__(self):
        import numpy as np
        import torch
        dataFolder = '/Volumes/mzxjm1_SSD/data/data/'
        self.bvals = torch.tensor(np.genfromtxt(dataFolder + 'bvals', dtype=np.float32))
        self.bvecs = torch.tensor(np.genfromtxt(dataFolder + 'bvecs', dtype=np.float32))

    def __call__(self, params):
        import numpy as np
        import torch
        params = params.flatten()
        n_fib = int((len(params) - 1) / 3)
        
        # if len(params) % 2 == 0:
        #     SNR = params[-1]
        #     n_fib = int((len(params) - 3) / 3)
        # else:
        #     SNR = 30
        #     n_fib = int((len(params) - 2) / 3)
        n_fib = int((len(params) - 2) / 3)
        SNR=30

        d = params[0]
        s0 = params[1]  # 1
        v = np.zeros((n_fib, 3))
        sumf = 0
        signal = torch.tensor((np.zeros((len(self.bvals)))))  # np.zeros((len(b)))

        for i in range(0, n_fib):
            # th = np.arccos(1-2*params[3 + 3 * i])
            # phi = 2*np.pi*params[4 + 3 * i]
            fi = params[2 + 3 * i]
            sumf += fi
            th = params[3 + 3 * i]
            phi = params[4 + 3 * i]
            v[i] = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
            # v = np.array([params[3 + 3 * i], params[4 + 3 * i], params[5 + 3 * i]])

            signal += s0 * (fi * np.exp(-d * self.bvals * np.power(np.dot(self.bvecs.T, v[i]), 2)))

        signal += s0 * (1 - sumf) * np.exp(-self.bvals * d)  # isotropic contribution
        sigma = np.abs(torch.mean(signal[np.nonzero(self.bvals)]) / SNR)
        noise = torch.empty(signal.shape).normal_(mean=0, std=sigma)
        signal = signal + noise

        def correct_signal(signal):
            signal = signal.numpy().ravel()
            outliers = np.argwhere(signal[5:] >= max(signal[:5])) + 5  # +5 because in the comparison we are excluding the first 5 values
            if len(outliers) > 0:
                corrected_val = min(signal[:5]) - max(signal[:5]) / 100
                signal[outliers] = corrected_val
            return signal

        signal = correct_signal(signal)
        signal = torch.Tensor(signal)

        return signal


class simulator_normalized:

    def __init__(self):
        import numpy as np
        import torch
        dataFolder = '/Volumes/mzxjm1_SSD/data/data/'
        self.bvals = torch.tensor(np.genfromtxt(dataFolder + 'bvals', dtype=np.float32))
        self.bvecs = torch.tensor(np.genfromtxt(dataFolder + 'bvecs', dtype=np.float32))

    def __call__(self, params):
        import numpy as np
        import torch
        params = params.flatten()

        # if len(params) % 2 == 0:
        #     SNR = params[-1]
        #     n_fib = int((len(params) - 3) / 3)
        # else:
        #     SNR = 30
        #     n_fib = int((len(params) - 2) / 3)
        n_fib = int((len(params) - 2) / 3)
        SNR=30

        d = params[0]
        s0 = params[1]  # 1
        v = np.zeros((n_fib, 3))
        sumf = 0
        signal = torch.tensor((np.zeros((len(self.bvals)))))  # np.zeros((len(b)))

        for i in range(0, n_fib):
            # th = np.arccos(1-2*params[3 + 3 * i])
            # phi = 2*np.pi*params[4 + 3 * i]
            fi = params[2 + 3 * i]
            sumf += fi
            th = params[3 + 3 * i]
            phi = params[4 + 3 * i]
            v[i] = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
            # v = np.array([params[3 + 3 * i], params[4 + 3 * i], params[5 + 3 * i]])

            signal += s0 * (fi * np.exp(-d * self.bvals * np.power(np.dot(self.bvecs.T, v[i]), 2)))

        signal += s0 * (1 - sumf) * np.exp(-self.bvals * d)  # isotropic contribution
        sigma = np.abs(torch.mean(signal[np.nonzero(self.bvals)]) / SNR)
        noise = torch.empty(signal.shape).normal_(mean=0, std=sigma)
        signal = signal + noise

        normalization = True
        signal_correct = True
        
        # normalization
        b0vols = signal[:5]
        mean_b0vols = torch.mean(b0vols)
        signal = signal / mean_b0vols

        
        def correct_signal(signal):
            signal = signal.numpy().ravel()
            outliers = np.argwhere(signal[5:] >= max(signal[:5])) + 5  # +5 because in the comparison we are excluding the first 5 values
            if len(outliers) > 0:
                corrected_val = min(signal[:5]) - max(signal[:5]) / 100
                signal[outliers] = corrected_val
            return signal

        signal = correct_signal(signal)
        signal = torch.Tensor(signal)

        return signal


class simulator_nos0:

    def __init__(self):
        import numpy as np
        import torch
        dataFolder = '/Volumes/mzxjm1_SSD/data/data/'
        self.bvals = torch.tensor(np.genfromtxt(dataFolder + 'bvals', dtype=np.float32))
        self.bvecs = torch.tensor(np.genfromtxt(dataFolder + 'bvecs', dtype=np.float32))

    def __call__(self, params):
        import numpy as np
        import torch
        params = params.flatten()

        # if len(params) % 2 == 0:
        #     SNR = params[-1]
        #     n_fib = int((len(params) - 2) / 3)
        # else:
        #     SNR = 30
        #     n_fib = int((len(params) - 1) / 3)
        n_fib = int((len(params) - 1) / 3)
        SNR=30

        d = params[0]
        s0 = 100
        v = np.zeros((n_fib, 3))
        sumf = 0
        signal = torch.tensor((np.zeros((len(self.bvals)))))  # np.zeros((len(b)))

        for i in range(0, n_fib):
            # th = np.arccos(1-2*params[3 + 3 * i])
            # phi = 2*np.pi*params[4 + 3 * i]
            fi = params[1 + 3 * i]
            sumf += fi
            th = params[2 + 3 * i]
            phi = params[3 + 3 * i]
            v[i] = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
            # v = np.array([params[3 + 3 * i], params[4 + 3 * i], params[5 + 3 * i]])

            signal += s0 * (fi * np.exp(-d * self.bvals * np.power(np.dot(self.bvecs.T, v[i]), 2)))

        signal += s0 * (1 - sumf) * np.exp(-self.bvals * d)  # isotropic contribution
        sigma = np.abs(torch.mean(signal[np.nonzero(self.bvals)]) / SNR)
        noise = torch.empty(signal.shape).normal_(mean=0, std=sigma)
        signal = signal + noise

        def correct_signal(signal):
            signal = signal.numpy().ravel()
            outliers = np.argwhere(signal[5:] >= max(signal[:5])) + 5  # +5 because in the comparison we are excluding the first 5 values
            if len(outliers) > 0:
                corrected_val = min(signal[:5]) - max(signal[:5]) / 100
                signal[outliers] = corrected_val
            return signal

        #signal = correct_signal(signal)
        #signal = torch.Tensor(signal)

        return signal


class simulator_nos0_norm:

    def __init__(self):
        import numpy as np
        import torch
        dataFolder = '/Volumes/mzxjm1_SSD/data/data/'
        self.bvals = torch.tensor(np.genfromtxt(dataFolder + 'bvals', dtype=np.float32))
        self.bvecs = torch.tensor(np.genfromtxt(dataFolder + 'bvecs', dtype=np.float32))

    def __call__(self, params):
        import numpy as np
        import torch
        params = params.flatten()

        # if len(params) % 2 == 0:
        #     SNR = params[-1]
        #     n_fib = int((len(params) - 2) / 3)
        # else:
        #     SNR = 30
        #     n_fib = int((len(params) - 1) / 3)
        
        n_fib = int((len(params) - 1) / 3)
        
        #SNR = params[-1]
        SNR=30
        d = params[0]
        s0 = 1
        v = np.zeros((n_fib, 3))
        sumf = 0
        signal = torch.tensor((np.zeros((len(self.bvals)))))  # np.zeros((len(b)))

        for i in range(0, n_fib):
            # th = np.arccos(1-2*params[3 + 3 * i])
            # phi = 2*np.pi*params[4 + 3 * i]
            fi = params[1 + 3 * i]
            sumf += fi
            th = params[2 + 3 * i]
            phi = params[3 + 3 * i]
            v[i] = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
            # v = np.array([params[3 + 3 * i], params[4 + 3 * i], params[5 + 3 * i]])

            signal += s0 * (fi * np.exp(-d * self.bvals * np.power(np.dot(self.bvecs.T, v[i]), 2)))

        signal += s0 * (1 - sumf) * np.exp(-self.bvals * d)  # isotropic contribution
        sigma = np.abs(torch.mean(signal[np.nonzero(self.bvals)]) / SNR)
        noise = torch.empty(signal.shape).normal_(mean=0, std=sigma)
        signal = signal + noise

        # normalization
        b0vols = signal[:5]
        mean_b0vols = torch.mean(b0vols)
        signal = signal / mean_b0vols
        signal = (signal - torch.min(signal)) / (torch.max(signal)- torch.min(signal))

        def correct_signal(signal):
            signal = signal.numpy().ravel()
            signal[signal<0] = 0.0001
            #signal[signal>1] = 1.0
            # And then correct the signal that is higher than b0 volumes
            outliers = np.argwhere(signal[5:] >= max(signal[:5])) + 5  # +5 because in the comparison we are excluding the first 5 values
            if len(outliers) > 0:
                corrected_val = min(signal[:5]) - max(signal[:5]) / 100
                signal[outliers] = corrected_val
            
            signal[signal>1] = 1.0
            return signal

        signal = correct_signal(signal)
        signal = torch.Tensor(signal)

        return signal
