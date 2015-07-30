"""
Main experiment runner.
"""

import argparse
import collections
import json
import os
import subprocess

import numpy as np

from experiments.plots import dim_sensitivity_plot


DIMS_RANGE = 10


def run_table():
    """
    Run the main experiment for populating the main results table.
    """

    split = '-s 0.2'
    n_iter= '--niter=10'
    experiments = ('stackexchange', 'movielens',)
    args = (('MFWarm', ['--ids']),
            ('MFCold', ['--ids', '--cold']),
            ('LightFMTagsWarm', ['--tags']),
            ('LightFMTagsCold', ['--tags', '--cold']),
            ('LightFMTagsIdsWarm', ['--tags', '--ids']),
            ('LightFMTagsIdsCold', ['--tags', '--ids', '--cold']),
            ('LightFMTagsAboutWarm', ['--tags', '--about']),
            ('LightFMTagsAboutCold', ['--tags', '--cold', '--about']),
            ('LightFMTagsAboutIdsWarm', ['--tags', '--ids', '--about']),
            ('LightFMTagsAboutIdsCold', ['--tags', '--ids', '--cold', '--about']),
            ('LSILRWarm', ['--tags', '--lsi']),
            ('LSILRCold', ['--tags', '--lsi', '--cold']),
            ('LSIUPWarm', ['--tags', '--up']),
            ('LSIUPCold', ['--tags', '--up', '--cold']))

    results = {}

    for experiment in experiments:
        filepath = os.path.join('experiments',
                                experiment,
                                'model.py')
        for name, options in args:

            if experiment == 'movielens' and 'About' in name:
                # This configuration is invalid for Movielens
                continue
            
            arglist = ['ipython', '--', filepath, split, n_iter] + options
            output = subprocess.check_output(arglist)

            # Take the value for default dimensionality.
            auc = json.loads(output).values()[0]
            results['%s%s' % (experiment.title(), name)] = auc

    # Embolden the top result
    for experiment in experiments:
        for scenario in ('Warm', 'Cold'):
            max_score = max(v for k, v in results.items() if scenario in k 
                            and experiment in k.lower())
            for name, auc in results.items():
                if scenario in name and experiment in name.lower():
                    if auc == max_score:
                        fmt_string = '\\newcommand\\var%s{\\textbf{\\fmtResult{%s}}}'
                    else:
                        fmt_string = '\\newcommand\\var%s{\\fmtResult{%s}}'

                    print(fmt_string % (name,
                                        str(auc)))


def run_dim_sensitivity():
    """
    Run experiments for producing latent dimension sensitivity plots.
    """

    dims = [x for x in range(2, DIMS_RANGE)]

    experiments = ('stackexchange', 'movielens',)
    split = '-s 0.2'
    n_iter = '--niter=30'

    args = (('LSI-UP', ['--tags', '--up', '--cold']),
            ('LSI-LR', ['--tags', '--lsi', '--cold']),
            ('LightFM (tags)', ['--tags', '--cold']),
            ('LightFM (tags + about)', ['--tags', '--cold', '--about']),
            ('LightFM (tags + ids)', ['--tags', '--ids', '--cold']),)

    results = {}

    for experiment in experiments:
        experiment_results = collections.defaultdict(list)

        filepath = os.path.join('experiments',
                                experiment,
                                'model.py')

        for name, options in args:
            exp_dims = [2**dim for dim in dims]
            arglist = (['ipython', '--', filepath, split]
                       + options
                       + [n_iter]
                       + ['--dim']
                       + [str(x) for x in exp_dims])
            output = json.loads(subprocess.check_output(arglist))
            experiment_results['%s' % name] += [output[str(x)] for x in exp_dims]

        results[experiment] = experiment_results

    # Save the results to a file.
    path = os.path.join('experiments', 'dim_sensitivity_results.json')
    with open(path, 'w') as results_file:
        results_file.write(json.dumps(results))


def draw_dim_sensitivity_plots():
    """
    Draw plots.
    """

    dims = [2**x for x in range(2, DIMS_RANGE)]

    # Get the results from a file
    path = os.path.join('experiments', 'dim_sensitivity_results.json')
    with open(path, 'r') as results_file:
        results = json.load(results_file)

    for experiment_name, experiment_results in results.items():

        # Draw the plot
        fname = os.path.join('paper', 'dim_sensitivity_%s.pdf' % experiment_name)
        dim_sensitivity_plot(np.array(dims),
                             experiment_results,
                             fname,
                             show_legend=(experiment_name == 'stackexchange'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('-t', '--table', action='store_true',
                        help='Run accuracy experiments for all models')
    parser.add_argument('-d', '--dim', action='store_true',
                        help='Run the latent dimensionality sensitivity experiment')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Create plots')
    args = parser.parse_args()

    if args.table:
        run_table()
    if args.dim:
        run_dim_sensitivity()
    if args.plot:
        draw_dim_sensitivity_plots()
