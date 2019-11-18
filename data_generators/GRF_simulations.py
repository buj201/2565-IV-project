import numpy as np
import pandas as pd

def sample_X(n,p):
    return np.random.normal(loc=0, scale=1, size=(n,p))

def sample_epsilon(n):
    return np.random.normal(loc=0, scale=1, size=n)

def sample_Z(n):
    return np.random.binomial(n=1,p=1/3,size=n)

def sample_Q(n, omega, epsilon):
    return np.random.binomial(n=1,p=1/(1+np.exp(-omega*epsilon)),size=n)
    
def get_W(Z,Q):
    return Z * Q

def get_tau(kappa, additive, p, X):
    assert kappa <= p, "Kappa assumed to be smaller than p." +\
                       f" You passed kappa={kappa} and p={p}".format(kappa,p)
    
    if additive:
        tau = np.sum(np.clip(X[:,0:kappa],a_min=None, a_max=0),axis=1)
    else:
        tau = np.clip(np.sum(X[:,0:kappa],axis=1),a_min=None, a_max=0)
    return tau
        
def get_mu(kappa, additive, p, X, nuisance_terms = [5,6], nuisance_scale=3):
    """
    nuisance_terms: 1-indexed to be in line with the GRF paper.
    """
    
    nuisance_terms = np.array(nuisance_terms) - 1
    
    assert nuisance_terms.min() >= kappa, "Nuisance terms overlap with effect terms."
    
    if additive:
        mu = np.sum(np.clip(X[:,nuisance_terms],a_min=None, a_max=0),axis=1)
    else:
        mu = np.clip(np.sum(X[:,nuisance_terms],axis=1),a_min=None, a_max=0)
        
    return nuisance_scale*mu
        
def get_Y(mu, W, tau, epsilon):
    return mu + (W - 1/2)*tau + epsilon

def get_sample(p, n, omega, kappa, additive, nuisance, seed):
    
    np.random.seed(seed)
    X = sample_X(n,p)
    epsilon = sample_epsilon(n)
    Q = sample_Q(n,omega,epsilon)
    Z = sample_Z(n)
    W = get_W(Z,Q)
    tau = get_tau(kappa, additive, p, X)
    mu = get_mu(kappa, additive, p, X)
    Y = get_Y(mu, W, tau, epsilon)
    
    return X, Y, W, Z, tau