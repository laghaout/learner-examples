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

    def wrangle(self, wrangler_class=wra.Wrangler) -> None:

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
        
    def train(self, part: str='train') -> None:

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

    def test(self, part: str='test') -> None:

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
        
        
    def test_report(self, average=None, print2screen=True) -> None:
               
        super().test_report()
                 
        labels = list(self.data.encoder.classes_)
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
        #     if self.env.containers.INSIDE_GCP in (True, 'Yes'):
        #         lesson_dir = os.path.join(*f'{self.env.paths.lesson_dir}'.split(os.path.sep)[2:])
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
            data = self.data.dataset[part]
        elif isinstance(data, list):
            mode = "online"
            print("Online serving.")
        elif isinstance(data, str) or isinstance(data, int):
            mode = "batch"
            print("Batch serving.")
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
            elif mode == "batch":       
                # fs = gcsfs.GCSFileSystem(project=self.env.cloud.PROJECT_ID)
                # data = pd.read_csv(
                #     fs.open(data), delimiter=self.data.delimiter)
                self.batch_prediction_job = self.serve_batch2bq(data)
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
                
        prediction_proba=pd.DataFrame(
            prediction_proba, columns=self.data.encoder.classes_)
        prediction_proba[self.data.target_name] = prediction
        
        self.report[part] = dict(
            data=data,
            prediction=prediction_proba,
            mode=mode,
            delta_tau=time.time() - delta_tau)
    
    def serve_report(self, part: str='serve'):
        
        super().serve_report()
        
        for k, v in self.report[part].items():
            print(f"{k}:")
            print(v)
            print()

    def serve_batch2bq(self, model_number, machine_type='n1-standard-4'):
    
        # Initialize connection
        aiplatform.init(location=self.env.cloud.REGION)
        
        # Get model that will make a batch prediction
        model_id = f"projects/{self.env.cloud.PROJECT_NUMBER}/locations/{self.env.cloud.REGION}/models/{model_number}"
        model = aiplatform.Model(model_id)
        
        # Check the supported batch prediction jobs input formats
        model.supported_input_storage_formats
        
        # Define required arguments for batch prediction job
        job_display_name = self.env.paths.dir_name
        bigquery_source = f'bq://{self.env.cloud.PROJECT_ID}.{self.env.paths.dir_name}.serve'
        bigquery_destination_prefix = f'bq://{self.env.cloud.PROJECT_ID}.{self.env.paths.dir_name}.predict'
        
        # Create batch prediction job
        batch_prediction_job = model.batch_predict(
            job_display_name=job_display_name,
            bigquery_source=bigquery_source,
            bigquery_destination_prefix=bigquery_destination_prefix,
            machine_type=machine_type
        )
        
        return batch_prediction_job

#%% Run as script, not as a module.
if __name__ == "__main__":

    learner = Learner(**kwargs)

    # If the keyword "serve" isn't part of the arguments, run the whole 
    # learner pipeline.
    if 'serve' not in sys.argv:
        learner(explore=True, select=False, train=False, test=False, serve=False)    
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
        if learner.env.containers.INSIDE_GCP in (True, 'Yes'):
            lesson_dir = os.path.join(*f'{lesson_dir}'.split(os.path.sep)[2:])
            fs = gcsfs.GCSFileSystem(project=learner.env.cloud.PROJECT_ID)
            learner_file = fs.open(f'{lesson_dir}/learner.pkl', 'rb')
        else:
            learner_file = open(f'{lesson_dir}/learner.pkl', 'rb')
        learner = pickle.load(learner_file)
        
        # Serve
        learner.serve(data)
        learner.serve_report()
