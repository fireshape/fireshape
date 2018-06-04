import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
linestyles = OrderedDict(
        [('solid',               (0, ())),
            ('loosely dotted',      (0, (1, 10))),
            ('dotted',              (0, (1, 5))),
            ('densely dotted',      (0, (1, 1))),

            ('loosely dashed',      (0, (5, 10))),
            ('dashed',              (0, (5, 5))),
            ('densely dashed',      (0, (5, 1))),

            ('loosely dashdotted',  (0, (3, 10, 1, 10))),
            ('dashdotted',          (0, (3, 5, 1, 5))),
            ('densely dashdotted',  (0, (3, 1, 1, 1))),

            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
ls = ['o', 's', '<', '>']
m = [(0.00, 0.1), (0.05, 0.1), (0.05, 0.1), (0.00, 0.1), (0.00, 0.1)]
labels = {
        'base_elasticity_cr_True': r'$CR(10^{-2})+\mathring{H}(\mathrm{sym})$',
        'base_elasticity_cr_False': r'$\mathring{H}(\mathrm{sym})$',
        'base_laplace_cr_True': r'$CR(10^{-2})+\mathring{H}^1$',
        'base_laplace_cr_False': r'$\mathring{H}^1$'
        }
m = [(0.04, 0.16), (0.08, 0.16), (0.12, 0.16), (0.00, 0.16)]
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 5,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }

plt.rcParams.update(params)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
f, axarr = plt.subplots(1, 2)
f.set_size_inches(11.4/2.54, 4/2.54)
for counter, name in enumerate([ 'base_elasticity_cr_False','base_elasticity_cr_True',
              'base_laplace_cr_False', 'base_laplace_cr_True']):
    d = "./output/" + name + "/"
    Jvals = np.load(d + "Jvals.npy")
    cnorms = np.load(d + "cnorms.npy")
    gnorms = np.load(d + "gnorms.npy")
    pde_solves = np.load(d + "pde_solves.npy")
    axarr[0].plot(pde_solves, Jvals, label=labels[name], marker=ls[counter],
                 markevery=m[counter], markersize=2, linewidth=0.5)
    axarr[1].semilogy(pde_solves, gnorms/gnorms[0], label=labels[name], marker=ls[counter],
                 markevery=m[counter], markersize=2, linewidth=0.5)
    # axarr[2].semilogy(pde_solves, cnorms, label=labels[name])

for ax in axarr:
    ax.set_xlabel("PDE Solves")

axarr[0].legend()
axarr[0].set_title('Functional value')
axarr[1].legend()
axarr[1].set_title('Relative norm of gradient of Lagrangian')
# axarr[2].legend()
# axarr[2].set_title('Constraint violation')
mydir = r'./output/img/'
plt.tight_layout(pad=0.)
f.savefig(mydir + "stokes_convergence.pdf", dpi=1000)
