#!/usr/bin/env python

VERSION = "0.1.1"
BUILD   = 2

#---------------------------------
# DRUGGS:  Identify drug-gene interactions in paired sample
#          genomic perturbation screens using Gibbs Sampling
# written by Medina Colic and Traver Hart
# Last modified 15 Jun 2018
# Free to modify and redistribute (MIT license)
#---------------------------------

import sys
import time

start = time.time()

from pylab import *

import pandas as pd
import numpy as np
import scipy.stats as stats


# ------------------------------------
# constants
norm_value  = 1e7
min_reads_thresh = 1
half_window_size = 500
# ------------------------------------


def druggs(readfile, druggs_outfile, control_samples, drug_samples, fc_outfile,
          remove_genes=None, pseudocount=5, minObs=1, index_column=0, verbose=False):
          ''' Normalize the raw read counts
              Apply emprirical bayes variance estimate of gRNA variances
              Gibbs sampling '''

    def log_(msg):
        if verbose:
            six.print_(msg, file=sys.stderr)

    num_replicates = len(control_samples)

    log_('Control samples:  ' + str(control_samples))
    log_('Treated samples:  ' + str(drug_samples))

    ###
    #read sgRNA reads counts file
    ###
    reads = pd.read_table(readfile, index_col=index_column)

    # remove control genes
    # e.g. TKOv1 genes ['chr10Promiscuous','chr10Rand','chr10','EGFP','LacZ','luciferase']
    # TKOv3: 'EGFP','LacZ','luciferase'
    if ( remove_genes ):
        reads = reads.loc[~reads[ reads.columns.values[0] ].isin(remove_genes),:]

    numGuides, numSamples = reads.shape

    ###
    #normalize to norm_value reads
    ###
    log_('Normalizing read counts')
    normed = norm_value * reads[control_samples+drug_samples] / reads[control_samples+drug_samples].sum().as_matrix()


    ###
    #Caculate fold change with normalized reads + pseudocount
    # maintain raw read counts for future filtering
    ###
    log_('Processing data')
    fc = pd.DataFrame(index=reads.index.values)
    fc['GENE'] = reads[ reads.columns.values[0] ]      # first column of input file MUST be gene name!

    for k in range(len(control_samples)):
        log_('Calculating raw fold change for replicate {0}'.format(k+1))
        fc[control_samples[k]] = reads[ control_samples[k] ]
        fc[drug_samples[k]] = reads[ drug_samples[k] ]
        fc['fc_{0}'.format(k)] = np.log2(( normed[ drug_samples[k] ] + pseudocount ) / ( normed[ control_samples[k] ]+ pseudocount))
        ###
        # sort guides by readcount, descending:
        ###
        fc.sort_values(control_samples[k], ascending=False, inplace=True)

        eb_std_samplid  = 'eb_std_{0}'.format(k)
        fc[eb_std_samplid] = np.zeros(numGuides)
        ###
        # get mean, std of fold changes based on 800 nearest fc
        ###
        log_('Caculating smoothed Epirical Bayes estimates of stdev for replicate {0}'.format(k+1))
        ###
        # initialize element at index 250
        ###
        # do not mean-center. fc of 0 should be z=score of 0.
        #ebmean = fc.iloc[0:500]['fc_{0}'.format(k)].mean()
        #fc[eb_mean_samplid][0:250] = ebmean


        ebstd  = fc.iloc[0:half_window_size*2]['fc_{0}'.format(k)].std()
        fc[eb_std_samplid][0:half_window_size]  = ebstd
        ###
        # from 250..(end-250), calculate mean/std, update if >= previous (monotone smoothing)
        ###
        for i in range(half_window_size, numGuides-half_window_size+25, 25):
            #every 25th guide, calculate stdev. binning/smoothing approach.
            ebstd  = fc.iloc[i-half_window_size:i+half_window_size]['fc_{0}'.format(k)].std()
            if (ebstd >= fc[eb_std_samplid][i-1]):
                fc[eb_std_samplid][i:i+25] = ebstd              #set new std in whole step size (25)
            else:
                fc[eb_std_samplid][i:i+25] = fc.iloc[i-1][eb_std_samplid]
        ###
        # set ebstd for bottom half-window set of guides
        ###
        #log_('Smoothing estimated std for replicate {0}'.format(k+1))
        fc[eb_std_samplid][numGuides-half_window_size:] = fc.iloc[numGuides-(half_window_size+1)][eb_std_samplid]
        ###
        # calc z score of guide
        ###
        log_('Caculating Zscores for replicate {0}'.format(k+1))
        fc['Zlog_fc_{0}'.format(k)] = fc['fc_{0}'.format(k)] / fc[eb_std_samplid]

    ###
    # write fc file as intermediate output
    ###
    if ( fc_outfile ):
    	fc.to_csv( fc_outfile, sep='\t', float_format='%4.3f')


    usedColumns = ['Zlog_fc_{0}'.format(i) for i in range(num_replicates)]



    ###
    # just processed fold change columns [Zlog_fc]
    ###
    data = fc[usedColumns]
    genes = list(set(fc.GENE))

    ###
    # store mean and std of sampled means and taus
    ###
    badger = pd.DataFrame(index=sorted(genes), columns=['mu_mean','mu_sig','tau_mean','tau_sig'])


    # define gibbs_sampling functions

    def mu_update(n, ybar, tau, u_prior, tau_prior):
        ''' Update mu values
            mu ~ N(mu_prior, sig2_prior)'''
        mean_numerator = ( (n*ybar*tau) + (u_prior*tau_prior) )
        mean_denominator= (n*tau) + tau_prior
        tau_term = ( n*tau + tau_prior )
        return stats.norm.rvs(loc=mean_numerator/mean_denominator, scale=1./sqrt(tau_term), size=1 )[0]


    def tau_update(n, data, mu, alpha_prior, beta_prior):
        ''' Update tau values
            tau ~ Gamma(alpha_prior, beta_prior)'''
        alpha_update = alpha_prior + n/2.
        sum_squared_dev = sum(( data-mu)**2)
        beta_update = beta_prior + sum_squared_dev/2
        return stats.gamma.rvs(alpha_update, scale=1./beta_update, size=1)[0]

    log_('Gibbs sampling started')

    def gibbs_sampler( data, numiter, u_prior, tau_prior, alpha_prior, beta_prior):
        ''' Sample for mu and tau '''
        ybar = data.mean()
        n = len(data)

        post_mu  = zeros(numiter)
        post_tau = zeros(numiter)

        ###
        # initialize mu_new, sig2_new
        ###
        mu_new = stats.norm.rvs(loc=u_prior, scale=sqrt(1./tau_prior), size=1)[0]
        tau_new = stats.gamma.rvs(alpha_prior, scale=1./beta_prior, size=1)[0]

        ###
        # run the chain, including burn-in, but skipping initial value
        ###
        for i in range(numiter):
            mu_new = mu_update(n, ybar, tau_new, u_prior, tau_prior)
            tau_new = tau_update( n, data, mu_new, alpha_prior, beta_prior)
            post_mu[i] = mu_new
            post_tau[i] = tau_new
        return post_mu, post_tau

    # set mu priors
    #u ~ N(u_prior, sig2_prior)
    u_prior = 0.
    tau_prior = 0.5  # weak prior: wide distribution for u


    # set tau weak priors here
    # tau ~ Gamma(alpha_prior, beta_prior)
    alpha_prior = 2.
    beta_prior = 1.



    ###
    # gibbs sampling of mu and tau for each gene
    ###

    log_('Writing output file')
    for g in sorted(genes):
        data = asarray(fc.iloc[ find(fc.GENE==g) ][['Zlog_fc_0','Zlog_fc_1', 'Zlog_fc_2']]).flatten()
        data = data[ isfinite( data) ]
        ybar = data.mean()
        n = len(data)
        mu_chain, tau_chain = gibbs_sampler(data, 1000, u_prior, tau_prior, alpha_prior, beta_prior)
        badger.loc[g]['mu_mean','mu_sig','tau_mean','tau_sig'] = [mu_chain.mean(), mu_chain.std(), \
                                                                  tau_chain.mean(), tau_chain.std()]


    fout = druggs_outfile
    if not hasattr(fout, 'write'):
        fout = open(fout, 'w')
    fout.write('GENE')
    cols = badger.columns.values
    for c in cols:
        fout.write('\t' + c)
    fout.write('\n')

    for i in badger.index.values:
        fout.write('{0:s}\t{1:4.3f}\t{2:4.3f}\t{3:4.3f}\t{4:4.3f}\n'.format(i,
                  badger.loc[i, 'mu_mean'], badger.loc[i, 'mu_sig'], badger.loc[i, 'tau_mean'], badger.loc[i, 'tau_sig']))

    fout.close()




def main():
    import argparse

    ''' Parse arguments. '''
    p = argparse.ArgumentParser(description='Gibbs sampling approach for detecting drug-gene interactions with CRISPR screens',epilog='dependencies: pylab, pandas, numpy, scipy')
    p._optionals.title = "Options"
    p.add_argument("-i", dest="infile", type=argparse.FileType('r'), metavar="sgRNA_count.txt", help="sgRNA readcount file", default=sys.stdin)
    p.add_argument("-o", dest="druggs", type=argparse.FileType('w'), metavar="druggs-output.txt", help="drugz output file", default=sys.stdout)
    p.add_argument("-f", dest="fc_outfile", type=argparse.FileType('w'), metavar="druggs-foldchange.txt", help="druggs normalized foldchange file (optional")
    p.add_argument("-c", dest="control_samples", metavar="control samples", required=True, help="control samples, comma delimited")
    p.add_argument("-x", dest="drug_samples", metavar="drug samples", required=True, help="treatment samples, comma delimited")
    p.add_argument("-r", dest="remove_genes", metavar="remove genes", help="genes to remove, comma delimited", default='')
    p.add_argument("-p", dest="pseudocount", type=int, metavar="pseudocount", help="pseudocount (default=5)", default=5)
    p.add_argument("-I", dest="index_column", type=int, help="Index column in the input file (default=0; GENE_CLONE column)", default=0)
    p.add_argument("--minobs", dest="minObs", type=int,metavar="minObs", help="min number of obs (default=6)", default=6)
    p.add_argument("-q", dest="quiet", action='store_true', default=False, help='Be quiet, do not print log messages')

    args = p.parse_args()

    control_samples = args.control_samples.split(',')
    drug_samples = args.drug_samples.split(',')
    remove_genes = args.remove_genes.split(',')

    if len(control_samples) != len(drug_samples):
        p.error("Must have the same number of control and drug samples")

    druggs(args.infile, args.druggs, control_samples, drug_samples,
    	args.fc_outfile, remove_genes, args.pseudocount, args.minObs, args.index_column, not args.quiet)

if __name__=="__main__":
    main()

end = time.time()
print(end - start)
