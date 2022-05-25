from typing import Dict, Text

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import attr

from metrics import evaluation_util
from common import keys

from model_interpreter import saliency_map
from data_loader import process_snp_data
from models import experiment_config
from models import dual_cnn
from metrics import callback_metrics

import tensorflow as tf
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './public_data/SoyNAM', 'Dirctory of dataset')
flags.DEFINE_string('result_dir', './results', 'Dirctory of output')
flags.DEFINE_string('dataset_type', 'height',
                    'Type of dataset (height|oil|moisture|protein|yield).')


def train_model():
  """train model
  """
  training_filename = os.path.join(FLAGS.data_dir,
                                   "%s.train.csv" % (FLAGS.dataset_type))
  test_filename = os.path.join(FLAGS.data_dir,
                               "%s.test.csv" % (FLAGS.dataset_type))

  dataset_dict = process_snp_data.load_snp_data_with_multi_labels(
      training_filename, ["label"], data_type='zygosity')

  x_other = dataset_dict[keys.KEY_FEATURES]
  y_other = dataset_dict[keys.KEY_LABEL]

  dataset_dict = process_snp_data.load_snp_data_with_multi_labels(
      test_filename, ["label"], data_type='zygosity')

  x_test = dataset_dict[keys.KEY_FEATURES]
  y_test = dataset_dict[keys.KEY_LABEL]

  # y_other = stats.zscore(y_other)
  x_train, x_valid, y_train, y_valid = train_test_split(x_other,
                                                        y_other,
                                                        test_size=len(x_test))

  # define model
  experimental_params = experiment_config.ExperimentalParams()
  model_hyperparams = dual_cnn.ModelHParamsDuallCNN()

  # model = model_deep_gwas.DeepGwasModel(model_hyperparams)
  model = dual_cnn.make_dual_cnn_model(x_train,
                                       x_train.shape[-1],
                                       model_hyperparams)
  model.compile(
      loss=experimental_params.loss,
      optimizer=experimental_params.optimizer,
      metrics=[experimental_params.metrics, callback_metrics.pearsonr])

  early_stopping = EarlyStopping(monitor='val_mae',
                                 patience=6,
                                 restore_best_weights=True)
  model.summary()

  # model training
  model.fit(x=x_train,
            y=y_train,
            batch_size=experimental_params.batch_size,
            epochs=experimental_params.epochs,
            validation_data=(x_valid, y_valid),
            verbose=1,
            callbacks=[early_stopping])

  # get predicted values for traning and validation dataset.
  pred_train_y = model.predict(x_train)
  pred_valid_y = model.predict(x_valid)
  pred_test_y = model.predict(x_test)

  # generate performance
  output_filename = os.path.join(FLAGS.result_dir,
                                 f'performance_{FLAGS.dataset_type}.csv')
  evaluation_util.save_performance(pred_train_y, y_train, pred_valid_y, y_valid,
                                   pred_test_y, y_test, output_filename)

  # generate saliency map
  output_filename = os.path.join(FLAGS.result_dir,
                                 f'saliency_{FLAGS.dataset_type}.png')
  saliency_data = saliency_map.run_saliency_map(model, x_test, output_filename)
  saliency_df = pd.DataFrame.from_records(saliency_data,
                                          columns=['index', 'saliency'])

  output_filename = os.path.join(FLAGS.result_dir,
                                 f'saliency_{FLAGS.dataset_type}_test.csv')
  saliency_df.to_csv(output_filename, index=False)


def main(_):

  tf.random.set_seed(1)
  np.random.seed(1)

  train_model()


if __name__ == "__main__":
  app.run(main)
