import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from IPython.display import display, Math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
np.seterr(divide='ignore', invalid='ignore')

# Naive Bayes come da corso UNITN
class NaiveBayesStandard:

    def __init__(self,soglia=0.26,alpha=1):
      self.soglia=soglia
      self.alpha=alpha

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape

        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self.P_spam_parole1=np.array([(np.sum(X[y==1,c])/np.sum(y)).ravel()[0] for c in range(X.shape[1])])
        self.P_spam_parole0 = 1-self.P_spam_parole1

        self.P_ham_parole1=np.array([(np.sum(X[y==0,c])/(len(y)-np.sum(y))).ravel()[0] for c in range(X.shape[1])])
        self.P_ham_parole0 = 1-self.P_ham_parole1

        self.P_spam=np.sum(y)/len(y)
        self.P_ham=1-self.P_spam

    def predict(self, X):
        self.spamworld=np.array([np.prod((self.P_spam_parole1*R) +((1-R)*self.P_spam_parole0)) for R in X])
        self.hamworld=np.array([np.prod((self.P_ham_parole1*R) +((1-R)*self.P_ham_parole0)) for R in X])
        posterior = (self.spamworld*self.P_spam)/(self.spamworld*self.P_spam+self.hamworld*self.P_ham)

        # implementazione metodo 2 UNITN: funziona solo con poche parole!! (lentissimo!)
        # sw=np.ones(X.shape[0])
        # hw=np.ones(X.shape[0])

        # for j in range(X.shape[0]):
        #   for i in range(X.shape[1]):
        #     sw[j] = sw[j] * self.P_spam_parole1[i] if (X[j,i]>=1) else self.P_spam_parole0[i]
        #     hw[j] = hw[j] * self.P_ham_parole1[i] if (X[j,i]>=1) else self.P_ham_parole0[i]
        # posterior = sw*self.P_spam/(sw*self.P_spam+hw*self.P_ham)

        classe = pd.Series(np.where(posterior>self.soglia,1,0),name="Posterior")
        return classe

# Naive Bayes come da corso UNITN variante calcolo con sommatoria di log
class NaiveBayesLog:

    def __init__(self,soglia=0.26):
      self.soglia=soglia

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self.P_spam_parole1=np.array([(np.sum(X[y==1,c])/np.sum(y)).ravel()[0] for c in range(X.shape[1])])
        self.P_spam_parole0 = 1-self.P_spam_parole1

        self.P_ham_parole1=np.array([(np.sum(X[y==0,c])/(len(y)-np.sum(y))).ravel()[0] for c in range(X.shape[1])])
        self.P_ham_parole0 = 1-self.P_ham_parole1

        self.P_spam=np.sum(y)/len(y)
        self.P_ham=1-self.P_spam

    def predict(self, X):
        #correzione alpha Laplace smoothing (x+alpha)/(y+alpha) --> alpha =1 quando y=0??
        sw=np.array([np.sum(np.log( (self.P_spam_parole1*R) +((1-R)*self.P_spam_parole0) )) for R in X])
        hw=np.array([np.sum(np.log((self.P_ham_parole1*R) +((1-R)*self.P_ham_parole0))) for R in X])

        alpha=np.array(np.maximum(sw.ravel(),hw.ravel()))
        self.spamworld=np.exp(sw+alpha)
        self.hamworld=np.exp(hw+alpha)
        posterior = (self.spamworld*self.P_spam)/(self.spamworld*self.P_spam+self.hamworld*self.P_ham)
        classe = pd.Series(np.where(posterior>self.soglia,1,0),name="Posterior")
        return classe
