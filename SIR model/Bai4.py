import pandas as pd
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import json
import datetime
from datetime import date


#1st mus, sigma : for beta
#2nd mus, sigma : for gamma 
mus = [0.66, 0.573]
sigma = [0.11, 0.1]

def gauss_pdf(param):
    return st.norm.pdf(param, mus, sigma)
     
def metropolis_hastings(iter=120):
    # Construct init parameter from normal pdf
    # 1st param: beta
    # 2nd param: gamma
    param = [0.6 , 0.5]
    pi = gauss_pdf(param)

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

        samples[i] = param

    return samples
#number=1580515200
if __name__ == '__main__':
    samples = metropolis_hastings(iter=100000)
    # for i in samples:
    #     print(i[0])
    number=1584835200	#number represents for date in timestamp (1584835200 = 22nd March, 2020)
    X =[]
    while number< 1595203200:  # 1595203200 = 20th July, 2020  
        timestamp = date.fromtimestamp(number)
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'+timestamp.strftime('%m-%d-%Y')+'.csv'
        df = pd.read_csv(url)
        confirmed = df['Confirmed'][df.Country_Region == 'Italy']
        recovered = df['Recovered'][df.Country_Region == 'Italy']
        x=confirmed+recovered
        X.append(x)
        number+=86400
    R0=0.0
    for x, sam in zip(X,samples):
        pi = st.gamma.pdf(x,a=sam[0],scale= sam[1])
        R0 = R0 + pi*(sam[0]/sam[1])
    print('R0 of Italy is ',sum(R0))

    while number< 1595203200:    # 1595203200 = 20th July, 2020  
        timestamp = date.fromtimestamp(number)
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'+timestamp.strftime('%m-%d-%Y')+'.csv'
        df = pd.read_csv(url)
        confirmed = df['Confirmed'][df.Country_Region == 'Vietnam']
        recovered = df['Recovered'][df.Country_Region == 'Vietnam']
        x=confirmed+recovered
        X.append(x)
        number+=86400
    R0=0.0
    for x, sam in zip(X,samples):
        pi = st.gamma.pdf(x,a=sam[0],scale= sam[1])
        R0 = R0 + pi*(sam[0]/sam[1])
    print('R0 of Vietnam is ',sum(R0))  