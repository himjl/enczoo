from enczoo.transforms.random_projection import RandomProjection
import torch
import time
import collections
import matplotlib.pyplot as plt
import numpy as np

xbudget = 1000000
p = 100


results = collections.defaultdict(list)

for d in [10, 100, 1000, 10000, 100000, 1000000]:
    b = max(1, xbudget // d)
    print('in shape', b, d)
    x = torch.randn((b, d))
    mod = RandomProjection(out_features=p, in_features=d, seed=0)

    niter = 5

    with torch.no_grad():
        samps = []
        for _ in range(niter):
            t0 = time.time()
            out = mod(x)
            t1 = time.time()
            samps.append(t1 - t0)

        mu = np.mean(samps)
        sem = np.std(samps) / np.sqrt(niter - 1)
        results['mu'].append(mu)
        results['sem'].append(sem)
        results['d'].append(d)

for k in results:
    results[k] = np.array(results[k])

# %%

plt.figure()
plt.errorbar(results['d'], results['mu'] * 1000, yerr=results['sem'] * 1000, marker='.')
plt.xscale('log')
plt.ylim([0, None])
#plt.yscale('log')
plt.xlabel('Feature dimension')
plt.ylabel('Time per projection (msec)')
plt.show()
