import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import json

# 1st mus, sigma : for beta
# 2nd mus, sigma : for gamma 

mu_beta = float(input("Insert mu for beta: "))
sigma_beta = float(input("Insert sigma for beta: "))
mu_gamma = float(input("Insert mu for gamma: "))
sigma_gamma = float(input("Insert sigma for gamma: "))
iter = int(input("Insert interation: "))

mus = [mu_beta, mu_gamma]
sigma = [sigma_beta, sigma_gamma]

def gauss_pdf(param):
    return st.norm.pdf(param, mus, sigma)
     
def metropolis_hastings(iter=1000):
    # Construct init parameter from normal pdf
    # 1st param: beta
    # 2nd param: gamma
    param = [0.6 , 0.5]
    pi = gauss_pdf(param)
    accepted = []

    # samples size is iterations, each sample is a pair (beta, gamma)
    samples = np.zeros((iter, 2))

    for i in range(iter):

        # Random normal distribution with size = 2 and new mean = previous beta, gamma
        param_star = np.array(param) + np.random.normal(size=2)
        
        # Calculate gauss_pdf with proposal sample 
        pi_star = gauss_pdf(param_star)

        # Calculate probability to keep the proposal sample
        r = min(1, (pi_star[0]*pi_star[1]*1.)/(pi[0]*pi[1]))

        # Generate random varible q from U(0,1)
        q = np.random.uniform(low=0, high=1)


        # if (q < r) keep proposal sample
        if q < r:
            param = param_star
            pi = pi_star
            accepted.append(param)

        samples[i] = param

    return samples, accepted



samples, accepted = metropolis_hastings(iter)
with open('samples2.txt', 'w') as f:
    f.writelines(','.join(str(j) for j in i) + '\n' for i in samples)

    
#print(accepted)
beta_hist = plt.figure()
sns.distplot(samples[:,0])
plt.xlabel(r'$\beta$')
plt.ylabel('Probability density')
beta_hist.savefig('beta2.png')

gamma_hist = plt.figure()
sns.distplot(samples[:,1])
plt.xlabel(r'$\gamma$')
plt.ylabel('Probability density')
gamma_hist.savefig('gamma2.png')

joint_distribution = plt.figure()
sns.jointplot(samples[:, 0], samples[:, 1])
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\gamma$')
plt.savefig('samples2.png')
    
