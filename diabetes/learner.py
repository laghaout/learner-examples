#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:26:41 2022

@author: ala
"""

import os
import tensorflow as tf
import learners.learner as lea

# Check whether the code is run from the Docker container. If so, there
# should not be any user name involved in the pathname to the data directory
# and the ``wrangler.py`` module should be imported with ``from .``.
try:
    from . import wrangler as wra
    # matplotlib.use('TkAgg')
    USER = ''
except BaseException:
    import getpass
    import wrangler as wra
    USER = getpass.getuser()

class Diabetes(lea.LearnerChild):

    def __init__(
            self,
            lesson_dir=['lessons'],
            data_params=dict(
                label_name='target',
                exploration_dir=['exploration'],
                shuffle=True,
            ),
            hyperparams=dict(
                early_stopping_threshold=.995,
                epochs=150,
                loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),  #'adam',
                metrics=['mae'],
            ),
            **kwargs):

        for k in ['exploration_dir']:
            if isinstance(data_params[k], list):
                data_params[k] = os.path.join(*data_params[k])

        super().__init__(
            lesson_dir=lesson_dir, data_params=data_params,
            hyperparams=hyperparams, **kwargs)
    
    def wrangle(self):
        
        super().wrangle()
        
        self.data = wra.Diabetes(**self.data_params)

    def explore(self):
        
        super().explore()
        
        print(self.data.dataset.head())
        self.report['explore'] = self.data.explore()

# %% Test outside the Docker container.

if len(USER) > 0:
    diabetes = Diabetes()
    diabetes()
    report = diabetes.report

