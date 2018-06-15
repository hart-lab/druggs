import sys

import time
from functools import wraps

start = time.time()

#def fn_timer(function):
#    @wraps(function)
#    def function_timer(*args, **kwargs):
#        t0 = time.time()
#        result = function(*args, **kwargs)
#        t1 = time.time()
#        print ("Total time running %s: %s seconds" %
#               (function.func_name, str(t1-t0))
#               )
#        return result
#    return function_timer



from pylab import *

import pandas as pd
import numpy as np
import scipy.stats as stats


def badger(readfile, index_column=0):
    # import data
    data = pd.read_table(readfile,  index_col=index_column)
    # extract just fold change columns
    fc = data[data.columns[[0,5,10,15]]]

    genes = list(set(fc.GENE))
    badger = pd.DataFrame(index=sorted(genes), columns=['mu_mean','mu_sig','tau_mean','tau_sig'])


    # define gibbs_sampling functions

    def mu_update(n, ybar, tau, u_prior, tau_prior):
        mean_numerator = ( (n*ybar*tau) + (u_prior*tau_prior) )
        mean_denominator= (n*tau) + tau_prior
        tau_term = ( n*tau + tau_prior )
        return stats.norm.rvs(loc=mean_numerator/mean_denominator, scale=1./sqrt(tau_term), size=1 )[0]


    def tau_update(n, data, mu, alpha_prior, beta_prior):
        alpha_update = alpha_prior + n/2.
        sum_squared_dev = sum(( data-mu)**2)
        beta_update = beta_prior + sum_squared_dev/2
        return stats.gamma.rvs(alpha_update, scale=1./beta_update, size=1)[0]

    def gibbs_sampler( data, numiter, u_prior, tau_prior, alpha_prior, beta_prior):
        ybar = data.mean()
        n = len(data)

        post_mu  = zeros(numiter)
        post_tau = zeros(numiter)

        #
        # initialize mu_new, sig2_new
        #
        mu_new = stats.norm.rvs(loc=u_prior, scale=sqrt(1./tau_prior), size=1)[0]
        tau_new = stats.gamma.rvs(alpha_prior, scale=1./beta_prior, size=1)[0]

        #
        # run the chain, including burn-in, but skipping initial value
            #
        for i in range(numiter):
            mu_new = mu_update(n, ybar, tau_new, u_prior, tau_prior)
    	    tau_new = tau_update( n, data, mu_new, alpha_prior, beta_prior)
            post_mu[i] = mu_new
            post_tau[i] = tau_new
        return post_mu, post_tau

    # set priors
    # u ~ N(u_prior, sig2_prior)
    u_prior = 0.
    tau_prior = 0.5  # weak prior: wide distribution for u

    # tau ~ Gamma(alpha_prior, beta_prior)
    # set weak priors here
    alpha_prior = 2.
    beta_prior = 1.

    for g in sorted(genes):
        data = asarray(fc.iloc[ find(fc.GENE==g) ][['Zlog_fc_0','Zlog_fc_1', 'Zlog_fc_2']]).flatten()
        data = data[ isfinite( data) ]
        ybar = data.mean()
        n = len(data)
        mu_chain, tau_chain = gibbs_sampler(data, 1000, u_prior, tau_prior, alpha_prior, beta_prior)
        badger.loc[g]['mu_mean','mu_sig','tau_mean','tau_sig'] = [mu_chain.mean(), mu_chain.std(), \
                                                                  tau_chain.mean(), tau_chain.std()]

    badger.to_csv("badgir_repABC_2.txt", sep="\t", float_format='%4.3f')


def main():
    import argparse

    ''' Parse args '''
    p = argparse.ArgumentParser(description='Gibbs sampler for chemogenetic interaction screens')
    p._optionals.title = "Options"
    p.add_argument("-i", dest="infile", type=argparse.FileType('r'), metavar="logZ_fold_change.txt", help="sgRNA logZ fold change file", default= sys.stdin)
    p.add_argument("-I", dest="index_column", type=int, help="Index column in the input file (default=0; GENE_CLONE column)", default=0)
    args = p.parse_args()

    badger(args.infile, args.index_column)

if __name__=="__main__":
    main()




end = time.time()
print(end - start)
