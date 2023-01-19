#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:57:26 2022

@author: ala
"""

import learners.learner as lea
import learners.utilities as util
import os
import tensorflow as tf
try:
    from . import wrangler as wra
except BaseException:
    import wrangler as wra

INSIDE_DOCKER_CONTAINER, USER, BASE_DIR = util.check_docker()

kwargs = dict(
    lesson_dir=['maps', 'domain', 'lessons'] if INSIDE_DOCKER_CONTAINER 
    else ['lessons'],
    data_params=dict(
        data_source=os.path.join(*BASE_DIR + ['Data', 'XXX.csv']), 
        nrows=None,
        delimiter='\t',
        index_col='XXX', 
        shuffle_seed=41, 
        parts={'train': .8, 'test': .2},
        fillna=0,
        target_name='XXX',
        ),
    hyperparams=dict(
        early_stopping_threshold=.9999,
        epochs=10,  # 100
        loss=tf.keras.losses.mae,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4, decay_steps=20000, decay_rate=0.9)),
        metrics=['mae'],
        hidden_units=[2**n for n in range(4, 2, -1)],
        activation='relu',
        output_activation='sigmoid',
        ),
    )

# %% Learner class
class Learner(lea.Learner):
    """ Generic domain learner """

    def __init__(self):
        
        super().__init__(**kwargs)
        
        assert True # Validate parameters if necessary.

    def wrangle(self, wrangler_class=wra.Wrangler):

        super().wrangle(wrangler_class)
        
#%% Learning outside the Docker container
if not INSIDE_DOCKER_CONTAINER:
    learner = Learner()
    learner(explore=True, select=True, train=True, test=True, serve=True)
    datasets = learner.data.datasets
    dataset = learner.data.dataset        