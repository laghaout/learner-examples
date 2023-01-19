#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:57:26 2022

@author: Amine Laghaout
"""

import learners.learner as lea
from learners.utilities import EnvManager
import learners.visualizer as vis
import os
import pandas as pd
import sklearn.metrics as sklmetrics
import tensorflow as tf
import time
try:
    from . import wrangler as wra
except BaseException:
    import wrangler as wra

# %% Configuration parameters

# Determine the environment parameters and the corresponding paths.
env = EnvManager(container_params=('INSIDE_DOCKER_CONTAINER', 'INSIDE_GCP'))
env(**dict(
    lesson_dir='lesson',   # Directory where to store the lesson.
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
    )

# %% Learner class
class Learner(lea.Learner):
    """ Generic domain learner """

    # def __init__(self):
        
    #     super().__init__(**kwargs)
        
    #     assert True # Validate parameters if necessary.

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
            self.report['test'] = dict(
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
            
        self.report['test']['delta_tau'] = time.time() - delta_tau
        
        
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

#%% Run as script, not as a module.
if __name__ == "__main__":

    env.summary()  # TODO: Delete
    # Learning outside the Docker container   
    if not env.containers.INSIDE_DOCKER_CONTAINER:
        learner = Learner(**kwargs)
        learner(explore=True, select=False, train=True, test=True, serve=False)
        datasets = learner.data.datasets
        dataset = learner.data.dataset        
        report = learner.report
    else:
        
        learner = Learner(**kwargs)
        learner(explore=True, select=False, train=True, test=True, serve=False)    
    env.summary()  # TODO: Delete

