#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:57:26 2022

@author: Amine Laghaout
"""

import gcsfs
from google.cloud import aiplatform
import learners.learner as lea
import learners.utilities as util
import learners.visualizer as vis
import numpy as np
import os
import pandas as pd
import pickle
import sklearn.metrics as sklmetrics
import sys
import tensorflow as tf
import time
try:
    from . import wrangler as wra
except BaseException:
    import wrangler as wra

# %% Configuration parameters

# Determine the environment parameters and the corresponding paths.
env = util.EnvManager(container_params=('INSIDE_DOCKER_CONTAINER', 'INSIDE_GCP'))
env(**dict(
    lesson_dir='lesson',    # Directory where to store the lesson.
    dir_name='iris',        # Name of the current directory.
    ))

kwargs = dict(
    lesson_dir=env.paths.lesson_dir,
    data_params=dict(
        data_source=os.path.join(
            env.paths.data_dir, f'{env.paths.dir_name}.csv'),
        nrows=None,
        delimiter='\t',
        shuffle_seed=41, 
        parts={'train': .8, 'test': .2},
        fillna=0,
        target_name='species',
        features=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        ),
    hyperparams=dict(
        early_stopping_threshold=.9999,
        epochs=40,  # 100
        batch_size=32,
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
        hidden_units=[512],
        output_units=3,
        activation='relu',
        output_activation='softmax',
        ),
    env=env,
    )

# %% Learner class
class Learner(lea.Learner):
    """ Iris classifier. """

    def wrangle(self, wrangler_class=wra.Wrangler):

        super().wrangle(wrangler_class)
        
    def design(self):
        
        super().design()
        
        early_stopping_threshold = self.hyperparams['early_stopping_threshold']   
        metric = self.hyperparams['metrics'][0]
        
        class EarlyStopping(tf.keras.callbacks.Callback):

            def on_epoch_end(
                    self,
                    epoch,
                    logs={},
                    early_stopping_threshold=early_stopping_threshold):

                if(logs.get(f"val_{metric}") > early_stopping_threshold):
                    print(
                        f'\nReached {early_stopping_threshold*100}%',
                        'validation accuracy so cancelling training!')
                    self.model.stop_training = True

        self.callbacks = [
            EarlyStopping(),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.lesson_dir, histogram_freq=1, profile_batch=0,
                write_images=True)]

        # Model
        self.model = tf.keras.Sequential(
            [tf.keras.layers.Dense(
                self.hyperparams['hidden_units'][0], 
                input_dim=len(self.data.features), 
                activation=self.hyperparams['activation'])]+
            [tf.keras.layers.Dense(
                hidden_units, activation=self.hyperparams['activation']) for 
             hidden_units in self.hyperparams['hidden_units'][1:]]+
            [tf.keras.layers.Dense(
                self.hyperparams['output_units'], 
                activation=self.hyperparams['output_activation'])])

        self.model.compile(
            loss=self.hyperparams['loss'],
            optimizer=self.hyperparams['optimizer'],
            metrics=self.hyperparams['metrics'])       
        
    def train(self, part: str='train'):

        super().train()

        delta_tau = time.time()

        self.model.fit(
            self.data.dataset[part][self.data.features], 
            # self.data.dataset[part][self.data.target_name],
            self.data.encoder.transform(self.data.dataset[part][self.data.target_name]),
            epochs=self.hyperparams['epochs'],
            # Use the same proportion for validation as for testing.
            validation_split=self.data_params['parts']['test'],
            callbacks=self.callbacks,
            verbose=self.verbose,
            batch_size=self.hyperparams['batch_size'],
            )
        
        self.report[part]['evaluation'] = self.model.evaluate(
            self.data.dataset[part][self.data.features],
            self.data.encoder.transform(self.data.dataset[part][self.data.target_name]),)
        
        self.report[part]['delta_tau'] = time.time() - delta_tau 

    def test(self, part: str='test'):

        super().test()

        delta_tau = time.time()
        
        if part in self.data.dataset.keys():
            self.report[part] = dict(
                zip(self.model.metrics_names,
                    self.model.evaluate(
                        self.data.dataset[part][self.data.features],
                        self.data.encoder.transform(self.data.dataset[part][self.data.target_name]),
                        verbose=self.verbose)))
            comparison = self.data.dataset[part][self.data.target_name].to_frame()
            predicted = self.model.predict(
                self.data.dataset[part][self.data.features])
            comparison['predicted'] = self.data.encoder.inverse_transform(predicted)
            comparison['error'] = comparison[self.data.target_name] != comparison.predicted
            self.report[part]['comparison'] = comparison             
        else:
            print(f"WARNING: There is no `{part}` data.")
            
        self.report[part]['delta_tau'] = time.time() - delta_tau
        
        
    def test_report(
            self,
            average=None,
            print2screen=True):
               
        super().test_report()
                 
        labels = list(learner.data.encoder.classes_)
        comparison = self.report['test']['comparison'].copy()
        comparison.drop('error', inplace=True, axis=1)
                   
        # Compute and plot the confusion matrix.        
        confusion_matrix = pd.DataFrame(
            sklmetrics.confusion_matrix(
                comparison[self.data.target_name], comparison['predicted'], 
                labels=labels),
            columns=pd.MultiIndex.from_tuples(
                [('predicted', label) for label in labels]),
            index=pd.MultiIndex.from_tuples(
                [(self.data.target_name, label) for label in labels]))
        
        vis.plot_confusion_matrix(
            confusion_matrix.values, labels, 
            save_as=os.path.join(self.lesson_dir, 'confusion_matrix.pdf')
            )
        
        # # General metrics
        metrics = pd.DataFrame(
            sklmetrics.classification_report(
                comparison[self.data.target_name], comparison['predicted'],
                output_dict=True))

        if print2screen:        
        #     # print("AUC:", AUC)
            print("\nMetrics:\n", metrics)
            print("\nConfusion matrix:\n", confusion_matrix)
        
        self.report['test'].update(
            dict(confusion_matrix=confusion_matrix,
                 # comparison_binarized=comparison, 
                  metrics=metrics, 
                 # AUC=AUC,
                 ))
        
    def serve(self, data=None, part: str='serve'):
        
        super().serve()
        
        delta_tau = time.time()

        #### Reload the model if it isn't attached to the learner.
        
        # TODO: Find a way to reload the model, both locally and on the GCP.
        # if 'model' not in self.__dict__.keys():
        #     if env.containers.INSIDE_GCP in (True, 'Yes'):
        #         lesson_dir = os.path.join(*f'{env.paths.lesson_dir}'.split(os.path.sep)[2:])
        #         pass
        #     else:
        #         lesson_dir = f'{self.env.paths.lesson_dir}'
        #         self.model = tf.keras.models.load_model(
        #             os.path.join(*[lesson_dir, 'model']))
        
        #### Determine whether we need online or batch prediction. 
        
        if data is None:
            mode = "in-memory"
            assert 'serve' in self.data.dataset.keys(), "No in-memory serving dataset"
            print("Test on `self.data.dataset['serve']`.")
            dataset = self.data.dataset[part]
        elif isinstance(data, list):
            mode = "online"
            print("Online serving.")
            dataset = data
        elif isinstance(data, str):
            mode = "batch"
            print("Batch serving.")
            dataset = util.rw_data(data)
        else:
            mode = "unknown"
            print("Unknown serving mode.")
        print()
        
        #### Perform the prediction.
                
        prediction_proba = None
        prediction = None
        
        # We are inside the GCP.
        if self.env.containers.INSIDE_GCP in (True, 'Yes'):
            
            print(f'Serving inside the GCP [mode: `{mode}`].')
            if mode == "online":
                endpoint = aiplatform.Endpoint(
                    endpoint_name=f"projects/{self.env.cloud.PROJECT_NUMBER}/locations/{self.env.cloud.REGION}/endpoints/{self.env.cloud.ENDPOINT_ID}")
                prediction_proba = endpoint.predict(instances=data).predictions
                prediction = self.data.encoder.inverse_transform(
                    np.array(prediction_proba))
            elif mode == "batch":       # TODO
                fs = gcsfs.GCSFileSystem(project=env.cloud.PROJECT_ID)
                dataset = pd.read_csv(fs.open(data), delimiter=self.data.delimiter)
                print(f"WARNING: Undetermined serving on the {mode} dataset.")
            elif mode == "in-memory":   # TODO
                print(f"WARNING: Undetermined serving on the {mode} dataset.")
            else:                       # TODO
                print(f"WARNING: Serving mode `{mode}`.")
        
        # We are not in the GCP.
        else:
            
            print('Serving outside of any container [mode: `{mode}`].')
            if mode == "online":        # TODO
                print(f"WARNING: Undetermined serving on the {mode} dataset.")
            elif mode == "batch":       # TODO
                print(f"WARNING: Undetermined serving on the {mode} dataset.")
            elif mode == "in-memory":   # TODO
                print(f"WARNING: Undetermined serving on the {mode} dataset.")
            else:                       # TODO
                print(f"WARNING: Serving mode `{mode}`.")
        
        #### Convert the numerical predictions into categories, if necessary.
        
        
        self.report[part]['delta_tau'] = time.time() - delta_tau
        self.report[part]['data'] = data
        
        return dict(
            prediction_proba=prediction_proba, 
            prediction=prediction,
            mode=mode)
      
#%% Run as script, not as a module.
if __name__ == "__main__":

    print('sys.argv:', sys.argv)  # TODO: Delete  
    learner = Learner(**kwargs)

    # Serve
    if 'serve' in sys.argv:
        
        data = [[5.7, 2.5, 5.0, 2.0], 
                [6.3, 3.3, 6.0, 2.5],
                [5.4, 3.9, 1.7, 0.4]]
        
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

    # Train
    else:

        learner(explore=True, select=False, train=True, test=True, serve=False)    
        learner.env.summary()  # TODO: Delete