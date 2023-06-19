#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:57:26 2022

@author: ala
"""

import learners.learner as lea
import learners.utilities as util
import os
import sys
import tensorflow as tf
try:
    from . import wrangler as wra
except BaseException:
    import wrangler as wra

# Determine the environment parameters and the corresponding paths.
env = util.EnvManager(
    container_params=('INSIDE_DOCKER_CONTAINER', 'INSIDE_GCP'))
env(**dict(
    lesson_dir='lesson',    # Directory where to store the lesson.
    dir_name='XXX',         # Name of the current directory.
    ))

kwargs = dict(
    lesson_dir=env.paths.lesson_dir,
    data_params=dict(
        data_source=os.path.join(env.paths.data_dir, f'{env.paths.dir_name}.csv'), 
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

    def wrangle(self, wrangler_class=wra.Wrangler) -> None:

        super().wrangle(wrangler_class)

#%% Run as script, not as a module.
if __name__ == "__main__":
    
    learner = Learner(**kwargs)

    # If the keyword "serve" isn't part of the arguments, run the whole 
    # learner pipeline.
    if 'serve' not in sys.argv:
        learner(explore=True, select=False, train=True, test=True, serve=False)    
        report = learner.report
        
    # If the first argument is "serve", use whatever comes next as the serving
    # data.
    elif 'serve' == sys.argv[1]:        
        assert len(sys.argv) >= 3
        if isinstance(sys.argv[2], str):
            try:
                data = eval(sys.argv[2])
            except BaseException:
                print('NOTE: The ``data`` passed is a string and should be',
                      'considered as the path to a BigQuery table.')

        # Retrieve the learner object, either from the GCP or locally.
        lesson_dir = learner.env.paths.lesson_dir
        print("<TODO: Serving should happen here>")
        # if learner.env.containers.INSIDE_GCP in (True, 'Yes'):
        #     lesson_dir = os.path.join(*f'{lesson_dir}'.split(os.path.sep)[2:])
        #     fs = gcsfs.GCSFileSystem(project=learner.env.cloud.PROJECT_ID)
        #     learner_file = fs.open(f'{lesson_dir}/learner.pkl', 'rb')
        # else:
        #     learner_file = open(f'{lesson_dir}/learner.pkl', 'rb')
        # learner = pickle.load(learner_file)
        
        # # Serve
        # learner.serve(data)
        # learner.serve_report()
    