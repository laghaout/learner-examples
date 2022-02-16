#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:32:52 2022

@author: ala
"""

from sklearn.datasets import load_diabetes

import learners.wrangler as wra
import learners.visualizer as vis

class Diabetes(wra.Wrangler):
    
    def acquire(self):
        
        self.dataset = load_diabetes(as_frame=True)
        self.dataset = self.dataset.frame
        if self.shuffle:
            self.dataset.sample(frac=1)
        
    def explore(self):
                
        corr = self.dataset.corr()
        columns = corr.sort_values(by=self.label_name).index
        
        report = {
            'description': self.dataset.describe(),
            'corr': corr.loc[columns, columns]}
        
        vis.plot_correlation_matrix(report['corr'], self.exploration_dir)
        vis.plot_pairgrid(
            self.dataset.iloc[:15], self.exploration_dir,
            )
        vis.plot_correlations(
            report['corr'], self.exploration_dir)
        
        return report
    
    
    


    
    
    