import pandas as pd
import os
import json
import math as m
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
import scipy
import random
import statsmodels.formula.api as smf 
import statsmodels.api as sm
    
def orderOfMagnitude(number):
    if number>0:
        return m.floor(m.log(number, 10))
    elif number==0:
        return 0
    else:
        return m.floor(m.log(-number, 10))
        
def roundP(p_value):
	# round funtion up to the order of magnitude except for 0.5 that round up to order of magnitud + 1
    if p_value!=0:
        p = np.round(p_value,-orderOfMagnitude(p_value))
        if p==0.5:
            p =  np.round(p_value,-orderOfMagnitude(p_value)+1)
        return p
    else:
        return p_value
    
def p_value_granger(p_value):
    # input: list with p values of the ACC
    # output: list with the first p_values less than 0.05 in list
    p_value = list(p_value)
    p_granger = []
    if any(elem<0.05 for elem in p_value):
        lastIndex2keep = np.array([np.where(np.array(p_value)>=0.05)]).min()-1
        for i in range(len(p_value)):
            if i<=lastIndex2keep:
                p_granger.append(p_value[i])
            else:
                p_granger.append(1)
    else:
        p_granger = np.ones(len(p_value))
    return p_granger
    
def Pearson_corr_coef(x,y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    return np.corrcoef(x[mask],y[mask])[0][1]

def Pearson_corr_coef2(x1,x2):
    return np.ma.corrcoef(np.ma.masked_invalid(x1),np.ma.masked_invalid(x2))[0][1]

def sem(x):
	# standard error of the mean
    sd = np.nanstd(x)
    n = sum(~np.isnan(x))
    return sd/np.sqrt(n)

# sigmoid function
def sigmoid(z): 
    return 1./(1+np.exp(-z))

def MSE_no(yRmean,yRfit):
	# mean squared error for non-optout psycometric fit
    return (np.dot((yRmean-yRfit),(yRmean-yRfit)))/6

def std_median(sample_list):
	# std median through bootstraping method
    n = len(sample_list)
    sampled_median = []
    for i in range(int(n/2)):
        sampled_list = random.sample(sample_list, int(n/2))
        sampled_median.append(np.nanmedian(sampled_list))
    return np.nanstd(sampled_median)

def cumul_norm(x,H,s):
    # cdf = (1/(s*sqrt(2*pi)))*int_-inf^x(exp(-((t-H)/s)^2/2)dt)
    return norm.cdf((x-H)/s)

def find_HR4(xhat,xstims,sigma,H,exp_points):
    epsilon = np.std(xstims)
    e2 = epsilon*epsilon
    s2 = sigma*sigma
    mu_ = (xhat+H)*e2/(s2+e2)
    s_ = np.sqrt(s2*e2/(s2+e2))
    ncdf = cumul_norm(0,mu_,s_)
    return (1-ncdf)/ncdf-exp_points

def pR4(xhat,xstims,sigma,H):
    epsilon = np.std(xstims)
    e2 = epsilon*epsilon
    s2 = sigma*sigma
    mu_ = (xhat+H)*e2/(s2+e2)
    s_ = np.sqrt(s2*e2/(s2+e2))
    ncdf = cumul_norm(0,mu_,s_)
    ctte = scipy.stats.norm(xhat,np.sqrt(e2+s2)).pdf(0)
    return ctte*(1-ncdf)

def pL4(xhat,xstims,sigma,H):
    epsilon = np.std(xstims)
    e2 = epsilon*epsilon
    s2 = sigma*sigma
    mu_ = (xhat+H)*e2/(s2+e2)
    s_ = np.sqrt(s2*e2/(s2+e2))
    ncdf = cumul_norm(0,mu_,s_)
    ctte = scipy.stats.norm(xhat,np.sqrt(e2+s2)).pdf(0)
    return ctte*(ncdf)

def llh_psycho_fit(theta,x,y_Rmean,y_Lmean,n):
    # Negative LogLikelihood of the multinomial distribution for dataset with optout
    # where prob of each selection is aproximated with normal cumulative distr
    # input:
    # theta: parameters [H_R,H_L,sigma]
    # H_R: rightward decision boundary 
    # H_L: leftward decision boundary (=H_R)
    # sigma: standard deviation of the internal response
    # x: array with stimuli values
    # y_Rmean: array with mean proportion of rightward answers over all the trials for each stimulus
    # y_Lmean: same as y_Rmean with leftward answers
    # n: total number of trials
    # output: negative log likelihood
    H_R = theta[0]
    H_L = theta[1]
    s = theta[2]
    ll1 = y_Rmean*n*np.log(cumul_norm(x,H_R,s))
    ll2 = y_Lmean*n*np.log(1.0-cumul_norm(x,H_L,s))
    ll3 = (1.0-y_Rmean-y_Lmean)*n*np.log(-cumul_norm(x,H_R,s)+cumul_norm(x,H_L,s))
    '''
    print(cumul_norm(xx,H_R,s),ll1)
    print(1.0-cumul_norm(xx,H_L,s),ll2)
    print(-cumul_norm(xx,H_R,s)+cumul_norm(xx,H_L,s),ll3)
    print('\n')
    '''
    return -np.sum(ll1+ll2+ll3)

def complete_llh(y_Rmean,y_Lmean,y_Rfit,y_Lfit,n):
    y_Omean = 1-y_Rmean-y_Lmean
    y_Ofit = 1 - y_Rfit-y_Lfit
    llf1,llf2,llf3,llf4,ll1,ll2,ll3 = 0,0,0,0,0,0,0
    for i in range(len(y_Rmean)):
        llf1 = llf1 + np.log(m.factorial(n))
        llf2 = llf2 + np.log(m.factorial(int(n*y_Rmean[i])))
        llf3 = llf3 + np.log(m.factorial(int(n*y_Lmean[i])))
        llf2 = llf2 + np.log(m.factorial(int(n*y_Omean[i])))
        ll1 = y_Rmean[i]*n*np.log(y_Rfit[i])
        ll2 = y_Lmean[i]*n*np.log(y_Lfit[i])
        ll3 = y_Omean[i]*n*np.log(y_Ofit[i])
    llh = llf1-llf2-llf3-llf4+ll1+ll2+ll3
    return llh

def fit_oo_binomial_LogReg(st,y):
    # Fit data with optout where the participant almost did not choose the
    # optout with logistic regression
    # input st: list with stimuli of each trial
    # inpuy y: array with the participant's answers (1 for rightward and 0 for left)
    # stimuli correspond to those trials where the participant did not choose the optout
    # output: array [H_R,H_L,sigma]
    # H_R: rightward decision boundary 
    # H_L: leftward decision boundary (=H_R)
    # sigma: standard deviation of the internal response
    # score: score fit in [0,1]
    n_total = len(st)
    # reshape of stimuli arrays for the logistic regression
    st = np.array(st).reshape(-1,1) 
    # add a column of ones to the stimuli arrays for the logistic regression
    X = np.ones((len(st),2))
    # second column of the stimuli for the logistic regression with the stimuli values
    for g in range(n_total):
        X[g][1] = st[g] 
    lr = LogisticRegression(C=1000000, fit_intercept=False)
    lr.fit(X,y)
    betas=lr.coef_[0]
    sigma = 1/betas[1]
    H_R = -betas[0]*sigma
    H_L = H_R
    score = np.round(lr.score(X, y),3)
    return [[H_R,H_L,sigma],score]

def fit_oo_binomial(st,y,data):
    # Fit data with optout where the participant almost did not choose the
    # optout with probit regression
    # input st: colum name of dataframe with stimuli of each trial
    # inpuy y: colum name of dataframe with the participant's answers (1 for rightward and 0 for left)
    # stimuli correspond to those trials where the participant did not choose the optout
    # data: dataframe
    # output: array [H_R,H_L,sigma]
    # H_R: rightward decision boundary 
    # H_L: leftward decision boundary (=H_R)
    # sigma: standard deviation of the internal response
    # pearson_chi2: 
    
    pb_fit = smf.glm(formula= y +'~'+ st, 
                family=sm.families.Binomial(link = sm.genmod.families.links.probit),
                data=data).fit()
    params = pb_fit.params
    H_R = -params[0]/params[1]
    H_L = H_R
    sigma = 1/params[1]
    pearson_chi2=pb_fit.pearson_chi2
    return [[H_R,H_L,sigma],pearson_chi2]

def fit_oo_multinomial(x,y_Rmean,y_Lmean,n):
    # Fit data with optout where the participant choose the optout with 
    # log likehood over the multinomial distribution of noise
    # input:
    # x: array with stimuli values
    # y_Rmean: array with mean proportion of rightward answers over all the trials for each stimulus
    # y_Lmean: same as y_Rmean with leftward answers
    # output: array [H_R,H_L,sigma]
    # H_R: rightward decision boundary 
    # H_L: leftward decision boundary
    # sigma: standard deviation of the internal response
    # success: 1.0 if the minimum neg log likelihood was found, 0.0 if not
    num_ini = 10
    ini_HR = np.linspace(0.001,0.1,num=num_ini)
    ini_HL = np.linspace(0.001,0.1,num=num_ini)
    ini_sigma = np.linspace(0.5,1.5,num=num_ini)
    # Find the parameters of the psychometric function that maximize 
    # the log likelihood
    llh_mat = np.zeros((num_ini,num_ini,num_ini))
    count1 = -1
    for hR in ini_HR:
        count1=count1+1
        count2 = -1
        for hL in ini_HL:
            count2=count2+1
            count3 = -1
            for sig in ini_sigma:
                count3=count3+1
                fitt = scipy.optimize.minimize(llh_psycho_fit,x0=[hR,-hL,sig],\
                                           args=(x,y_Rmean,y_Lmean,n),\
                                        options={'maxiter':3})
                params=fitt.x
                llh_mat[count1][count2][count3]=fitt.fun
    index=np.where(llh_mat==np.nanmin(llh_mat))
    print(np.nanmin(llh_mat))
    print('index: ',index)
    # best initial conditions over the parameters
    initial_HR=ini_HR[index[0][0]]
    initial_HL=ini_HL[index[1][0]]
    initial_s=ini_sigma[index[2][0]]
    fiteo = scipy.optimize.minimize(llh_psycho_fit,x0=[initial_HR,-initial_HL,initial_s],\
                   args=(x,y_Rmean,y_Lmean,n),method='BFGS')
    success = fiteo.success
    params = fiteo.x
    print(fiteo.message)
    return params,success

def MSE_oo(yRmean,yLmean,yOmean,yRfit,yLfit,yOfit):
    return (np.dot((yRmean-yRfit),(yRmean-yRfit))+np.dot((yLmean-yLfit),(yLmean-yLfit))+np.dot((yOmean-yOfit),(yOmean-yOfit)))/6

def Wrandom_fit_oo_multinomial(x,y_Rmean,y_Lmean,y_Omean,n):
    # Fit data with optout where the participant choose the optout with 
    # log likehood over the multinomial distribution of noise
    # input:
    # x: array with stimuli values
    # y_Rmean: array with mean proportion of rightward answers over all the trials for each stimulus
    # y_Lmean: same as y_Rmean with leftward answers
    # output: array [H_R,H_L,sigma]
    # H_R: rightward decision boundary 
    # H_L: leftward decision boundary
    # sigma: standard deviation of the internal response
    # success: 1.0 if the minimum neg log likelihood was found, 0.0 if not
    max_iter = 1000
    attempt = 0
    found = False
    mse_current,mse_old = 1000,1000
    while not found:
        attempt += 1
        ini_HR = random.uniform(0.00002,0.2)
        ini_HL = random.uniform(0.00002,0.2)
        ini_sigma = random.uniform(0.002,0.2)
        fiteo = scipy.optimize.minimize(llh_psycho_fit,x0=[ini_HR,-ini_HL,ini_sigma],\
                   args=(x,y_Rmean,y_Lmean,n),method='BFGS',options={'maxiter':1000})
        success = fiteo.success
        params = fiteo.x
        message = fiteo.message
        HR = params[0]
        HL = params[1]
        sigma = params[2]
        yR_fit = cumul_norm(x,HR,sigma)
        yL_fit = 1.0-cumul_norm(x,HL,sigma)
        yO_fit = 1.0-yR_fit-yL_fit
        
        mse = MSE_oo(y_Rmean,y_Lmean,y_Omean,yR_fit,yL_fit,yO_fit)   
        if mse<0.04:
            params_fit = params
            found = True
        else:
            if mse<mse_current:
                mse_old = mse_current
                mse_current = mse
                params_fit = [HR,HL,sigma]
            found = False
        if attempt == max_iter:
            #print('maximum iterations reached. best mse = '+str(mse_current))
            break
    return params_fit,found

def symmetrize(a):
	# create a symmetric matrix from array a, i.e. a = [1,2,3,4,5,6]
	# output: [[0. 1. 2. 3.], [1. 0. 4. 5.], [2. 4. 0. 6.], [3. 5. 6. 0.]]
	# output dimension = N, array a must have dimension N*(N-1)/2 
	a = np.array(a)
	l = a.shape[0]
	dim = int((1+np.sqrt(1+8*l))/2) # matrix dimension
	for_split = []
	for i in range(dim-2):
	    for_split.append(int((i+1)*(dim-1)-i*(i+1)/2))
	A = []
	x = np.split(a,for_split)
	x.append([])
	for i in range(dim):   
	    A.append(np.concatenate((np.zeros(i+1),x[i])))  
	lower = np.tril(np.transpose(A)) # lower part of the matrix
	return (A+lower)

class ReturnValue:
	# class for Linear_Regr function return
    def __init__(self, slope, intercept, r_value, p_value, std_err):
        self.slope = slope
        self.intercept = intercept
        self.r_value = r_value
        self.p_value = p_value
        self.std_err = std_err
    def __repr__(self):
        return "<slope:%s intercept:%s r_value:%s p_value:%s std_err:%s>" % (self.slope, self.intercept,self.r_value,self.p_value,self.std_err)
        
def Linear_Regr(x,y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask],y[mask])
    return ReturnValue(slope, intercept, r_value, p_value, std_err)

def fun_autocorr(x):
    cc = []
    x = list(x)
    n = len(x)
    for k in range(n-int(n/2)):
        x1 = x[:n-k]
        x2 = x[k:n]
        x1 = np.array(x1)
        x2 = np.array(x2)
        cc.append(np.ma.corrcoef(np.ma.masked_invalid(x1),np.ma.masked_invalid(x2))[0][1])
    return np.array(cc)

def mean_acc_shuffled(array,N):
	# mean autocorrelation of N random shuffled samples of array
    autocorr = []
    a = np.array(array)
    for i in range(N):
        np.random.shuffle(a)
        result = fun_autocorr(a)
        autocorr.append(result)
    mean_autocorr = np.nanmean(autocorr, axis=0)
    return mean_autocorr

def finalACC(x,mean_ap_x):
    # input:
    # - x: np.array with the temporal series (mood, food, ...) por each participant
    # - mean_ap_x: np.array with the mean x(i) across participants
    # output:
    # - acc_x_wn: autocorrelation of x for each subject
    X = x-mean_ap_x
    acc_x = fun_autocorr(X)
    mean_acc_x = mean_acc_shuffled(X,100)
    acc_x_wn = acc_x-mean_acc_x # acc without noise
    return acc_x_wn

def fun_autocorr_lag(x,lag):
# the same as fun_autocorr but k runs until lag
	cc = []
	x = list(x)
	n = len(x)
	for k in range(lag+1):
		x1 = x[:n-k]
		x2 = x[k:n]
		x1 = np.array(x1)
		x2 = np.array(x2)
		cc.append(np.ma.corrcoef(np.ma.masked_invalid(x1),np.ma.masked_invalid(x2))[0][1])
	return np.array(cc)

def mean_acc_shuffled_lag(array,N,lag):
	# same as mean_acc_shuffled but k runs until lag
    autocorr = []
    a = np.array(array)
    for i in range(N):
        np.random.shuffle(a)
        result = fun_autocorr_lag(a,lag)
        autocorr.append(result)
    mean_autocorr = np.nanmean(autocorr, axis=0)
    return mean_autocorr

def finalACC_lag(x,mean_ap_x,lag):
    # same as finalACC but k runs until lag
    X = x-mean_ap_x
    acc_x = fun_autocorr_lag(X,lag)
    mean_acc_x = mean_acc_shuffled_lag(X,100,lag)
    acc_x_wn = acc_x-mean_acc_x # acc without noise
    return acc_x_wn

class ReturnValueACC:
	# class for mean_sem_p_ACC function return
    def __init__(self, mean, se, p_value, pgranger):
        self.mean = mean
        self.se = se
        self.p_value = p_value
        self.pgranger = pgranger
    def __repr__(self):
        return "<mean:%s se:%s p_value:%s pgranger:%s>" % (self.mean, self.se,self.p_value,self.pgranger)
    
def mean_sem_p_ACC(MatrixAcc):
    # input: matrix with the finalACC for each participant as rows
    # output: mean, sem autocorrelation across subjects and p_value against the null hypotesis of zero mean
    # warning: p_value length is len(mean)+1 because we save the 
    mean_ACC = np.nanmean(MatrixAcc, axis=0)
    transp_ACC = np.transpose(MatrixAcc)
    len_ = [np.isnan(elem)[np.isnan(elem) == False].size for elem in transp_ACC]
    sd_ACC = np.nanstd(MatrixAcc, axis=0)
    se_ACC = np.array([sd_ACC[i]/np.sqrt(len_[i]) for i in range(len(len_))])
    p_ACC = np.array([stats.ttest_1samp(elem,0,axis=0)[1] for elem in transp_ACC])
    p_ACC = np.delete(p_ACC,0)
    pgranger = p_value_granger(p_ACC)
    mean_ACC = np.delete(mean_ACC,0)
    se_ACC = np.delete(se_ACC,0)

    return ReturnValueACC(mean_ACC, se_ACC, p_ACC, pgranger)

class ReturnValueCC:
	# class for mean_sem_p_ACC function return
    def __init__(self, mean, se, p_value):
        self.mean = mean
        self.se = se
        self.p_value = p_value
    def __repr__(self):
        return "<mean:%s se:%s p_value:%s>" % (self.mean, self.se,self.p_value)

def mean_sem_p_CC(MatrixAcc):
    # input: matrix with the finalACC for each participant as rows
    # output: mean, sem autocorrelation across subjects and p_value against the null hypotesis of zero mean
    # warning: p_value length is len(mean)+1 because we save the 
    mean_ACC = np.nanmean(MatrixAcc, axis=0)
    transp_ACC = np.transpose(MatrixAcc)
    len_ = [np.isnan(elem)[np.isnan(elem) == False].size for elem in transp_ACC]
    sd_ACC = np.nanstd(MatrixAcc, axis=0)
    se_ACC = np.array([sd_ACC[i]/np.sqrt(len_[i]) for i in range(len(len_))])
    p_ACC = np.array([stats.ttest_1samp(elem,0,axis=0)[1] for elem in transp_ACC])

    return ReturnValueCC(mean_ACC, se_ACC, p_ACC)

def fun_crosscorr(x,y):
    # x and y must have the same dimension
    x = list(x)
    n = len(x)
    cc1,cc2 = [],[]
    for k in range(n-int(n/2)):
        x1 = x[:n-k]
        x2 = y[k:n]
        x3 = x[k:n]
        x4 = y[:n-k]
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        x4 = np.array(x4)
        cc1.append(np.ma.corrcoef(np.ma.masked_invalid(x1),np.ma.masked_invalid(x2))[0][1])
        if k>0:
            cc2.append(np.ma.corrcoef(np.ma.masked_invalid(x3),np.ma.masked_invalid(x4))[0][1])   
    cc2 = cc2[::-1]
    cc = cc2+cc1
    return np.array(cc)

def cross_corr_mean_shuffled(array1,array2,N):
    crosscorr = []
    a1 = np.array(array1)
    a2 = np.array(array2)
    for i in range(N):
        np.random.shuffle(a2)
        result = fun_crosscorr(a1,a2)
        crosscorr.append(result)
    mean_crosscorr = np.mean(crosscorr, axis=0)
    return mean_crosscorr

def finalCC(x,mean_ap_x,y,mean_ap_y):
    # input:
    # - x,y: np.array with the temporal series (mood, food, ...) por each participant
    # - mean_ap_x,mean_ap_y: np.array with the mean x(i),y(i) across participants
    # output:
    # - cc_wn: crosscorrelation btw X and Y for each subject
    X = x-mean_ap_x
    Y = y-mean_ap_y
    cc = fun_crosscorr(X,Y)
    mean_cc = cross_corr_mean_shuffled(X,Y,100)
    cc_wn = cc - mean_cc # acc without noise
    return cc_wn

def fun_crosscorr_lag(x,y,lag):
	# same as fun_crosscorr  but k runs until lag
    cc = []
    x = list(x)
    n = len(x)
    for k in range(lag+1):
        x1 = x[:n-k]
        x2 = y[k:n]
        x1 = np.array(x1)
        x2 = np.array(x2)
        cc.append(np.ma.corrcoef(np.ma.masked_invalid(x1),np.ma.masked_invalid(x2))[0][1])
    return np.array(cc)

def cross_corr_mean_shuffled_lag(array1,array2,N,lag):
	# same as cross_corr_mean_shuffled  but k runs until lag
    crosscorr = []
    a1 = np.array(array1)
    a2 = np.array(array2)
    for i in range(N):
        np.random.shuffle(a2)
        result = fun_crosscorr_lag(a1,a2,lag)
        crosscorr.append(result)
    mean_crosscorr = np.mean(crosscorr, axis=0)
    return mean_crosscorr

def finalCC_lag(x,mean_ap_x,y,mean_ap_y,lag):
    # same as finalCC  but k runs until lag
    X = x-mean_ap_x
    Y = y-mean_ap_y
    cc = fun_crosscorr_lag(X,Y,lag)
    mean_cc = cross_corr_mean_shuffled_lag(X,Y,100,lag)
    cc_wn = cc - mean_cc # acc without noise
    return cc_wn

def p_stars(p):
    if p<0.05 and  p>=0.01:
        stars = '*'
    elif p<0.01 and p>=0.001:
        stars = '**'
    elif p<0.001 and p>=0.0001:
        stars = '***'
    elif p<0.0001:
        stars = '****'  
    else:
    	stars = ''
    return stars

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# not tested
def Pearson_cross_corr(x,y):
    cc = []
    x = list(x)
    for k in range(len(x)):
        xx = x[k::] + x[:k:]
        xx = np.array(xx)
        y = np.array(y)
        cc.append(np.ma.corrcoef(np.ma.masked_invalid(xx),np.ma.masked_invalid(y))[0][1])
    return np.array(cc)

def acc_LR(x):
    cc = []
    x = list(x)
    n = len(x)
    for k in range(n-2):
        x1 = x[:n-k]
        x2 = x[k:n]
        x1 = np.array(x1)
        x2 = np.array(x2)
        slope, intercept, r, p, se = stats.linregress(x1, x2)
        cc.append(r)
    return np.array(cc)    

def mean_acc_LR_shuffled(array,N):
    autocorr = []
    a = np.array(array)
    for i in range(N):
        np.random.shuffle(a)
        result = acc_LR(a)
        autocorr.append(result)
    mean_autocorr = np.mean(autocorr, axis=0)
    return mean_autocorr

def after_up2median(x):
    x = np.array(x)
    median_x = np.nanmedian(x)
    equal_median_sum = np.sum(x==median_x)
    y = []
    if equal_median_sum==0:
        for elem in x:
            if elem>median_x:
                y.append(np.nan)
            else:
                y.append(elem)  
    elif equal_median_sum==x.size/2:
        for elem in x:
            y.append(np.nan)
        print('must erase this participant who almost always answered the same')
    else:
        for elem in x:
            y.append(np.nan)
        print('analyze in detail this particular case, instead do not write a rule foe that')      
    return np.array(y)

def AIC(k,ll):
    return 2*k - 2*ll

def dAIC(k,ll,control_ll):
    return 2*(k-2) - 2*ll + 2*control_ll