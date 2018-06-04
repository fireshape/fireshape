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
labels = {
        'base_elasticity_cr_True': r'$CR(10^{-2})+\mathring{H}(\mathrm{sym})$',
        'base_elasticity_cr_False': r'$\mathring{H}(\mathrm{sym})$',
        'base_laplace_cr_True': r'$CR(10^{-2})+\mathring{H}^1$',
        'base_laplace_cr_False': r'$\mathring{H}^1$'
        }
ls = [list(linestyles.values())[i] for i in [0, 4, 5, 8, 9]]
ls = ['o', 's', '<', '>']
m = [(0.00, 0.1), (0.05, 0.1), (0.05, 0.1), (0.00, 0.1), (0.00, 0.1)]
plt.figure(figsize=(8/2.54, 5/2.54))
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 7,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
mydir = r'./output/img/'
counter = 0
qs = [1.5, 1.75, 2.0, 2.5, 3.0]
print("&" + '&'.join(str(q) for q in qs) + r"&\text{max}\\\hline")
for f in sorted(os.listdir(mydir)):
    if f.endswith(".npy"):
        label = labels[f[8:-4]]
        xy = np.load(mydir + f)
        qual = xy[:, 1]
        plt.plot(qual, xy[:, 2], label=label, marker=ls[counter],
                 markevery=m[counter], markersize=2, linewidth=0.5)
        print(label + "&"
              + "&".join(f"{100.*np.sum(qual>q)/len(qual):.2f}\%" for q in qs)
              + "&" + f"{np.max(qual):.2f}" + r"\\")
        counter += 1
        if counter == 4:
            qual = xy[:, 0]
            plt.plot(qual, xy[:, 2], label="Initial mesh", linewidth=0.5)
            print("Initial mesh" + "&" +
                  "&".join(f"{100.*np.sum(qual>q)/len(qual):.2f}\%" for q in qs)
                  + "&" + f"{np.max(qual):.2f}" + r"\\")
            counter += 1
plt.xlim((1, 1.5))
plt.xlabel(r"$\eta$")
plt.ylabel(r"Fraction of cells with $\eta(K)\le \eta$")
plt.legend()
plt.title("Mesh quality inverse CDF")
plt.tight_layout(pad=0.)
plt.savefig(mydir + "stokes_inv_cdf.pdf", dpi=300)
