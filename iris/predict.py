#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:57:26 2023

@author: Amine Laghaout
"""

from google.cloud import aiplatform
import gcsfs, pickle
import numpy as np
import pandas as pd

# %% Retrieve the cloud parameters.

try:
    from learners.utilities import EnvManager
    env = EnvManager(container_params=('INSIDE_DOCKER_CONTAINER', 'INSIDE_GCP'))
    env(**dict(
        lesson_dir='lesson',    # Directory where to store the lesson.
        dir_name='iris',        # Name of the current directory.
        ))
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

print('PROJECT_NUMBER', env.cloud.PROJECT_NUMBER)
print('ENDPOINT_ID', env.cloud.ENDPOINT_ID)
assert isinstance(env.cloud.PROJECT_NUMBER, int) and isinstance(env.cloud.ENDPOINT_ID, int)
print(f'PROJECT_NUMBER: {env.cloud.PROJECT_NUMBER}')
print(f'ENDPOINT_ID: {env.cloud.ENDPOINT_ID}')
    
# %% Serve

def serve(x_test, env):

    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{env.cloud.PROJECT_NUMBER}/locations/us-central1/endpoints/{env.cloud.ENDPOINT_ID}")
    
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
for k in fs.ls(f'{env.cloud.BUCKET}/entity/{env.paths.dir_name}/{env.paths.lesson_dir}'):
    print(k)

try:
    from learner import Learner 
    learner = Learner()
    learner = pickle.load(fs.open(f'{env.cloud.BUCKET}/entity/{env.paths.dir_name}/{env.paths.lesson_dir}/learner.pkl', 'rb'))
    print('- classes:')
    print(learner.data.encoder.classes_)
    print('- inverse transform:')
    print(learner.data.encoder.inverse_transform(np.array(prediction)))
except BaseException:
    print('ERROR: Failed to import the learner.')