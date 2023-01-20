#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:57:26 2023

@author: Amine Laghaout
"""

from learner import Learner
import learners.utilities as util
import gcsfs
import pickle
import os

data = [[5.7, 2.5, 5.0, 2.0], 
        [6.3, 3.3, 6.0, 2.5],
        [5.4, 3.9, 1.7, 0.4]]

env = util.EnvManager(container_params=('INSIDE_DOCKER_CONTAINER', 'INSIDE_GCP'))
env(**dict(
    lesson_dir='lesson',    # Directory where to store the lesson.
    dir_name='iris',        # Name of the current directory.
    ))

learner = Learner()

if env.containers.INSIDE_GCP in (True, 'Yes'):
    lesson_dir = os.path.join(
        *f'{env.paths.lesson_dir}'.split(os.path.sep)[2:])
    fs = gcsfs.GCSFileSystem(project=env.cloud.PROJECT_ID)
    learner = pickle.load(fs.open(f'{lesson_dir}/learner.pkl', 'rb'))
else:
    lesson_dir = env.paths.lesson_dir
    learner = pickle.load(open(f'{lesson_dir}/learner.pkl', 'rb'))

output = learner.serve(data)
print('output:')
print(output)

# %% Old
if False:

    from google.cloud import aiplatform
    import gcsfs, pickle
    import numpy as np
    import os
    import pandas as pd
    
    # %% Retrieve the cloud parameters.
    
    try:
        from learners.utilities import EnvManager
        env = EnvManager(container_params=('INSIDE_DOCKER_CONTAINER', 'INSIDE_GCP'))
        env(**dict(
            lesson_dir='lesson',    # Directory where to store the lesson.
            dir_name='iris',        # Name of the current directory.
            ))
        env.summary()
    except BaseException:
        print('WARNING: Could not load the cloud parameters automatically.', 
              'Reverting to the hard-coded parameters.')
        # TODO: FILL WITH THE ACTUAL VALUES
        from types import SimpleNamespace
        env = dict(cloud=dict(
            ENDPOINT_ID = int,      
            BUCKET = str,
            PROJECT_NUMBER = int,
            PROJECT_NAME = str))
        env['cloud'] = SimpleNamespace(**env['cloud'])
        env = SimpleNamespace(**env)
    
    # %% Serve
    
    def serve(x_test, env):
    
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{env.cloud.PROJECT_NUMBER}/locations/{env.cloud.REGION}/endpoints/{env.cloud.ENDPOINT_ID}")
        
        return endpoint.predict(instances=x_test).predictions
    
    # Prediction
    x_test = [[5.7, 2.5, 5.0, 2.0], 
              [6.3, 3.3, 6.0, 2.5],
              [5.4, 3.9, 1.7, 0.4]]
    prediction = serve(x_test, env)
    print('- predictions: ')
    print(prediction)
    
    # Retrieve the dataset
    fs = gcsfs.GCSFileSystem(project=env.cloud.PROJECT_ID)
    dataset = pd.read_csv(fs.open(f'{env.cloud.BUCKET}/entity/{env.paths.dir_name}/{env.paths.dir_name}.csv'), delimiter='\t')
    print(dataset.head())
    
    # List the model lesson file
    lesson_dir = os.path.join(*f'{env.paths.lesson_dir}'.split('/')[2:])
    for k in fs.ls(lesson_dir):
        print(k)
    
    from learner import Learner 
    learner = Learner()
    learner = pickle.load(fs.open(f'{lesson_dir}/learner.pkl', 'rb'))
    print('- classes:')
    print(learner.data.encoder.classes_)
    print('- inverse transform:')
    print(learner.data.encoder.inverse_transform(np.array(prediction)))
