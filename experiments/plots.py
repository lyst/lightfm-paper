import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

## import seaborn as sns
## sns.set_palette('Set1')
## sns.set_style('white')




FONTSIZE = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = FONTSIZE

DASHES = ['-', '--', '-.', ':']
MARKERS = ['.', '^', 'v', 'x', '+']

KEYS = ('LSI-LR',
        'LSI-UP',
        'LightFM (tags)',
        'LightFM (tags + ids)',
        'LightFM (tags + about)')

COLORS = ('#e41a1c',
          '#377eb8',
          '#4daf4a',
          '#984ea3',
          '#ff7f00')


def dim_sensitivity_plot(x, Y, fname, show_legend=True):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure(figsize=(3, 3))
    plt.xlabel('$d$', size=FONTSIZE)
    plt.ylabel('ROC AUC', size=FONTSIZE)

    plt.set_cmap('Set2')

    lines = []
    for i, label in enumerate(KEYS):
        line_data = Y.get(label)

        if line_data is None:
            continue
        
        line, = plt.plot(x, line_data, label=label, marker=MARKERS[i],
                         markersize=0.5 * FONTSIZE, color=COLORS[i])
        lines.append(line)



    if show_legend:
        plt.legend(handles=lines)
        plt.legend(loc='lower right')
    plt.xscale('log', basex=2)
    plt.xticks(x, [str(y) for y in x], size=FONTSIZE)
    plt.yticks(size=FONTSIZE)
    plt.tight_layout()

    plt.savefig(fname)
