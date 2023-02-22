#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:09:38 2022

@author: Amine Laghaout
"""

import pandas as pd
import learners.wrangler as wra
import learners.utilities as util

class Wrangler(wra.Wrangler):
    
    def acquire(self):
        
        super().acquire()
        
        self.dataset = pd.read_csv(self.data_source, delimiter=self.delimiter)
        
        print(self.dataset.head())  

    def __call__(self):
        """ Consolidate the fine-grained beliefs into a scalar target. """
        
        report = super().__call__()
        
        self.datasets['raw'] = self.dataset.copy()       
        
        from sklearn.preprocessing import LabelBinarizer
        self.encoder = LabelBinarizer()
        self.encoder.fit(list(set(self.dataset[self.target_name])))
        
        return report

    def explore(self, part: str='train'):
        
        report = super().explore()
        
        print(self.dataset[part].info())
        
        return report
        
    def split(self, split_sizes=None):

        super().split(split_sizes)

        # Split all the recorded datasets.
        if isinstance(self.datasets, dict):
            for k in self.datasets.keys():
                self.datasets[k] = util.split(self.datasets[k], self.parts)

        # Split the main dataset.
        self.dataset = util.split(self.dataset, self.parts)         
        
class OneOff:

    def __init__(self, **kwargs):

        import seaborn as sns
        import learners.utilities as util
        
        util.args_to_attributes(self, **kwargs)
        
        iris = sns.load_dataset('iris')
        iris = iris.sample(frac=1, random_state=41)
        iris.to_csv('data/data.csv', sep='\t', index=False)      