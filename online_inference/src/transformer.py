from numpy import mean
from numpy import std
from numpy import where
from numpy import logical_or
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, X, n_stdev):
        self.X = X
        self.n_stdev = n_stdev

    def fit(self, df, y=None):
        return self

    def transform(self, verbose=False):
        '''check for outliers and replace it by mean'''
        result = self.X.copy()
        for i in range(result.shape[1]):
            col = self.X.iloc[:, i]
            mu, sigma = mean(col), std(col)
            lower, upper = mu-(sigma*self.n_stdev), mu+(sigma*self.n_stdev)
            ix = where(logical_or(col < lower, col > upper))[0]
            if verbose and len(ix) > 0:
                print('>col=%d, outliers=%d' % (i, len(ix)))
            result.iloc[ix, i] = mu
        return result