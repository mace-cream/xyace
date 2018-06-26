#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:41:44 2016

@author: lizhong
"""

import numpy as np
import matplotlib.pyplot as plt


class XYACE:
    '''
    An object has the following attributes
    
    K = number of feature functions
    
    nIter  = the number of iterations to do ACE
    
    selbar = a random choice of bootstrapping, each iteration takes samples with 
             probability of selbar
    
    XAlpbet
    YAlpbet  = dictionary of items (index, value)
               XAlpbet[k] for k in range(len(XAlpbet)) gives the corresp value
               all internal calculation is based on index, which does not have
               empty values
    
    XAlpbetR
    YAlpbetR  = reverse dictioary, in case we need to get the index quick
                 XAlpbet[value] gives the corresponding index without searching
                 
    f, g    = functions of X and Y index from ACE, after the runACE() method is 
              run in __init__
              
    XInx, YInx  = the data samples translated into indices
    
    Px, Py  = the empirical distribution of the data in indices
    
    
    Methods
    
    GetPx(L), GetPy(L)
    GetF(L), GetG(L)  = in each case need argument as a list of values 
                        (not index), and return the list of results after using
                        the index translation. Ideally, this should be the only 
                        way to access the ACE results
    
    '''
    
    def __init__ (self, X, Y, K, OutlierControl=False):
    

        # count the number of sample pairs
        self.nSamples=min(len(X), len(Y))
        
        # counting to get the alphabet in a dictionary, and the empirical dist.
        # Also store in self.XInx and self.YInx the indices
        
        # XAlpbet[inx]=data, XAlpbetR[data]=inx
        
        (self.XAlpbet, self.XAlpbetR, self.Px, self.XInx)= self.enum(X)        
        (self.YAlpbet, self.YAlpbetR, self.Py, self.YInx)= self.enum(Y)
        
        '''
        # for debug only, create B matrix in the index order
        
        self.B = np.zeros(B.shape)
        
        for i in range(len(self.XAlpbet)):
            for j in range(len(self.YAlpbet)):
                self.B[j,i]= B[self.YAlpbet[j], self.XAlpbet[i]]
        
        XYCounts=np.zeros(B.shape)
        
        for n in range(self.nSamples):
            x=self.XInx[n]
            y=self.YInx[n]
            XYCounts[y,x]=XYCounts[y,x]+1
        
        self.Bemp=XYCounts/self.nSamples
        
        for i in range(len(self.XAlpbet)):
            for j in range(len(self.YAlpbet)):
                self.Bemp[j,i]=self.Bemp[j,i]/np.sqrt(self.Px[i]*self.Py[j])
        
        #checked Bemp = self.B

        '''
        
        # choose dimension of features chosen
        self.K= K
        self.OutlierControl=OutlierControl
        
        # random choice of f and g
        self.f=np.random.normal(0, 1, (len(self.XAlpbet), self.K))
        self.g=np.random.normal(0, 1, (len(self.YAlpbet), self.K))
        
        # choose number of iterations as 10 for now. 
        self.nIter=40
        
        # choose a way of picking data samples in each round, 
        # selbar=0.1 means each iteration takes only 10% of the entire data
        # randomly, when debugging set as 1
        self.selbar =1
        

        
        # run the ACE algorithm
        self.runACE()

    # end of __init__()    
    
    def enum(self, Data):
    # counting to get the alphabet in a dictionary, and the empirical dist    
    
        # initialize an empty alphabet  
        # in both directions      
        Alphabet={}         #Alphabet[index] = value
        AlphabetR={}        #AlphabetR[value] = index
        
        # initialize an empty empirical dist
        P=[]
        
        # store index in INX, one index per sample
        INX=[0]*len(Data)
        
        # go through all samples
        for i in range(self.nSamples):
            # if new symbol append Alphabet
            d=Data[i]
            if d not in AlphabetR:
                Alphabet[len(AlphabetR)]=d
                AlphabetR[d]=len(AlphabetR)
                P.append(0)
            # count occrance    
            P[AlphabetR[d]] = P[AlphabetR[d]]+1
            INX[i]= AlphabetR[d]
         
        # normalize P
        for i in range(len(P)):
            P[i]=float(P[i])/self.nSamples
            
        return(Alphabet, AlphabetR, np.asarray(P), INX) 
    # end of enum()   
        
    def runACE(self):
        # go thru all iterations
        for Iter in range(self.nIter):
            
            # beginning of the f(x)=E[g(Y)|X=x] step, first clear up self.f
            self.f=np.zeros(self.f.shape)
            
            # reset a counter for X in this iteration, may check if close to 
            # self.Px can ignore.
            counterX=[0]*len(self.XAlpbet)
            
            # go thru all samples
            for n in range(self.nSamples):
                
                # discard some samples 
                if np.random.random()<self.selbar:
                    x=self.XInx[n]
                    y=self.YInx[n]
                    counterX[x] = counterX[x]+1
                    self.f[x, :] = self.f[x,:] + self.g[y,:]
                
            for x in range(len(self.XAlpbet)):
                self.f[x,:]=self.f[x,:]/counterX[x]
            
            '''
            debug here
            
            
            phi=self.g[:,0]*np.sqrt(self.Py)
            psi=np.dot(np.transpose(self.B), phi)
            fhope=psi/np.sqrt(self.Px)
            psiemp=np.dot(np.transpose(self.Bemp), phi)
            femp=psiemp/np.sqrt(self.Px)
            print('iteration', Iter, 'fhope', fhope)
            print('empirical average', femp)
            print('actual calculation', self.f[:,0])
            
            
            end of debug
            '''
            
            '''
            Add the outlier control
            for each signature, of f[i,:], if the norm is too large, 
            we need to stop it from growing more. 
            
            Too large is defined as larger than self.K*1.5
            
            scall all these signatures by 1/2
            
            '''
            if self.OutlierControl:
                sqnorm=np.sum(self.f * self.f, axis=1)
                outliers=np.where(sqnorm>self.K*1.5)[0]
                # the last [0] is one of those strange things about np.where
                #print('Iteration', Iter,'---', len(outliers), 'outliers \n')
                self.f[outliers,:]=self.f[outliers,:]/3

                
            
            # beginning of the g(y)=E[f(X)|Y=y] step
            
            # reset g as zero float
            self.g=np.zeros(self.g.shape)
            
            # reset a counter for Y in this iteration
            counterY=[0]*len(self.YAlpbet)
            
            # go thru all samples
            for n in range(self.nSamples):
                if np.random.random()<self.selbar:
                    x=self.XInx[n]
                    y=self.YInx[n]
                    counterY[y]=counterY[y]+1
                    self.g[y,:]=self.g[y,:]+ self.f[x,:]
            
            for y in range(len(self.YAlpbet)):
                self.g[y,:] = self.g[y,:]/counterY[y]
                
            # begnning of Gram-Schmidt
            
            for i in range(self.K):
                mean=sum(self.g[:,i] *self.Py)
                self.g[:,i]=self.g[:,i]-mean
                for j in range(i):
                    # inner product between g[:,i] and g[:,j]
                    inij=sum(self.Py * self.g[:,i] * self.g[:,j])
                    self.g[:, i] = self.g[:,i] - inij * self.g[:,j]
                sqnorm=sum(self.Py * self.g[:,i] * self.g[:,i])
                self.g[:,i]= self.g[:,i] / np.sqrt(sqnorm)
                
    
    # end of runACE()
    
    def GetPx(self, L):
        return([ self.Px[self.XAlpbetR[x]] for x in L])
        
    def GetPy(self, L):
        return( [self.Py[self.YAlpbetR[y]] for y in L])
        
    def GetF(self, L):  # this returns a |L|x k array
        return(np.asarray( [self.f[self.XAlpbetR[x], :] for x in L]))
        
    def GetG(self, L):
        return(np.asarray( [self.g[self.YAlpbetR[y], :] for y in L]))
        
        
# end of class XYACE        

def normalize(v):
    return(v/np.sqrt(sum(v*v)))

def CheckSubspace(v, U, orthonormal=True):
    '''    
    for input vector v, write it as linear combinations of column vectors
    in U, and return the squared norm of the orthogonal element
    
    
    '''
    Y=U    
    if not orthonormal:  #Gram-Schmidt
       for i in range(U.shape[1]):
           for j in range(i):
               Y[:,i]= Y[:,i] - sum(Y[:,i]*Y[:,j])*Y[:,j]
               
           s=np.sqrt(sum(Y[:,i]*Y[:,i]))
           Y[:,i]=Y[:,i]/s

    proj=np.zeros(v.shape)
       
    for i in range(U.shape[1]):
        proj=proj + sum(v * Y[:,i])*Y[:,i]

    orth=v-proj
    
    return(sum(orth*orth)/sum(v*v))     
    # return ration of the squared norm
    
# end of CheckSubspace()
    
#
#if __name__ == "__main__":
#    
#    # the main function here is a test of the ACE algorithm
#    # We generate X in [1,2,3]
#    # and Y has distribution P1, P2, P3
#    # then samples of (X,Y) pairs, 
#    # With ACE, and K=2, we should be able to get two functions log P2/P1 
#    # log P3/P1, or any linear combination between the two. 
#    
#    #debugging of the __init__    
#    #X=[4, 2, 3, 4, 2, 3, 3, 1, 2, 3, 2, 1, 1, 2, 2, 3, 1, 4]    
#    #Y=[7,7,5,6,6,7,8,8,6,4,5,6,7,7,7,7,4,5,8,5,6]
#    #a=XYACE(X,Y)
#    
# 
#    
#    
#    Q=np.random.random((8,3))
#    CDFY=np.zeros(Q.shape)
#   
#    Px=[.25,.25,.5]   
#    CDFX=[.25, .5, 1] # cdf of X  
#    Py=[0.0]*8
#    
#    for i in range(3):
#        s=sum(Q[:, i])
#        Q[:,i] = Q[:,i]/s
#        Py=Py+ Q[:,i]*Px[i]
#    
#    epsilon=0.5
#
#    for i in range(3):
#        Q[:,i]= Py + epsilon*(Q[:,i]-Py)            
#    
#        
#    for i in range(3):
#        s=sum(Q[:, i])
#        for j in range(8):
#            CDFY[j,i]=sum(Q[:j+1, i])/s
#
#    BTrue=np.zeros(Q.shape)
#    for i in range(3):
#        for j in range(8):
#            BTrue[j,i]= Q[j,i]*np.sqrt(Px[i]/Py[j])    
#    
#    nSamples=10000
#    X=[0]*nSamples
#    Y=[0]*nSamples
#    
#    for n in range (nSamples):
#        x=np.random.random()
#        X[n]= sum([x>p for p in CDFX])
#        y=np.random.random()
#        Y[n]= sum([y>p for p in CDFY[:,X[n]]])
#        
#        
#    ''' 
#    check empirical distribution
#    
#    XYEmp=np.zeros(Q.shape)
#    
#    for n in range(nSamples):
#        XYEmp[Y[n], X[n]] = XYEmp[Y[n], X[n]] +1
#        
#    XYEmp=XYEmp/nSamples
#    XYDesire=Q*Px
#    
#    '''
#    
#    a=XYACE(X,Y)  
#   
#    
#   
#   
#    # check the results
#    #LLR1 = \log (P_{Y|X}(y|1)/P_{Y|X}(y|0)     
#    #LLR2 = \log (P_{Y|X}(y|2)/P_{Y|X}(y|0)      
#    # a.g[:,0] and a.g[:,1] should span the same subspace as LLR1 and LLR2 
#   
#    LLR1 = np.log(Q[:,1]/Q[:,0])
#    LLR2 = np.log(Q[:,2]/Q[:,0])
#
#    aPx = a.GetPx(range(3))
#    aPy = a.GetPy(range(8))
#    
##    plt.plot(range(8), aPy)
##    plt.show()
#    
#    B=np.zeros(Q.shape)
#    for i in range(3):
#        for j in range(8):
#            B[j,i]= Q[j,i]*np.sqrt(aPx[i]/aPy[j])
#            
#    U,s,V = np.linalg.svd(B, full_matrices=False)
#    G1 = U[:,1]/np.sqrt(aPy)
#    G2 = U[:,2]/np.sqrt(aPy)
#    
#    
#    print('Check SVD of B with LLR1', CheckSubspace(LLR1*np.sqrt(aPy), U[:, 1:3]))
#    print('Check SVD of B with LLR2', CheckSubspace(LLR2*np.sqrt(aPy), U[:, 1:3]))
#    
#
#    G= a.GetG(range(8))
#    G[:,0]=G[:,0]*np.sqrt(aPy)
#    G[:,1]=G[:,1]*np.sqrt(aPy)
#    
#    print('Check ACE with LLR1', CheckSubspace(LLR1*np.sqrt(aPy), G))
#    print('Check ACE with LLR2', CheckSubspace(LLR2*np.sqrt(aPy), G)) 
#
#        
#        
#    
    