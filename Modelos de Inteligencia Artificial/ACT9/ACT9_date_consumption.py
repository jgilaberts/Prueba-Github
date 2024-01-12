import numpy as np
import pytorch_lightning as pl
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import TSFEDL.models_keras as TSFEDL
import matplotlib.pyplot as plt
import torch
import csv
import os
import glob
from sklearn.preprocessing import StandardScaler
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--train", help="Do training? (default, false)", action='store_true')
argParser.add_argument("--n_input", help="Number of timesteps for input data")
argParser.add_argument("--n_output", help="Number of timesteps for output data")
argParser.add_argument("--batch_size", help="Batch_size", default=32)
argParser.add_argument("--n_epochs", help="Number of maximum epochs to run the deep learning models", default=150)
argParser.add_argument("--clase", help="Name of the target variable in the dataset",  default=".B03 Consumption kWh")

args = argParser.parse_args()
print(args)

# May limit GPU memory to 4GB
#gpus = tf.config.list_physical_devices('GPU')
#tf.config.set_logical_device_configuration(gpus[0],  [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 4)])


class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None, batch_size=32,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    self.batch_size = batch_size

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
       labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
      inputs, labels = self.example
      fig = plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col}')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
          label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
          label_col_index = plot_col_index

        if label_col_index is None:
          continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
          predictions = model(inputs)
          plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)

        if n == 0:
          plt.legend()

      plt.xlabel('Time [h]')
      plt.show()

      return fig



  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size, )

    ds = ds.map(self.split_window)

    return ds

    #WindowGenerator.make_dataset = make_dataset

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


def compile_and_fit(model, window, MAX_EPOCHS, patience=2, model_name="best_model.h5"):
    """
    It performs the training of the given model
    Parameters
    ----------
    model : Keras model
        Model to compile and fit.
    window : WindowGenerator instance
        Instance of the class WindowGenerator to create windows for training.
    MAX_EPOCHS : int
        Number of epochs for the training phase.
    patience : int
        Maximum number of epochs for the early stopping callback.
    Returns
    -------
    history : dict
        Dictionary with the history data of the model when fitted.
    """
    # We use early stopping and model checkpoint to store the vest model found so far.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    if not os.path.exists(f'./{INPUT_LENGTH}_{OUTPUT_STEPS}'):
        os.mkdir(f'./{INPUT_LENGTH}_{OUTPUT_STEPS}')

    # Only train if there is no previous training checkpoint file
    history = None
    if not os.path.exists(f'./{INPUT_LENGTH}_{OUTPUT_STEPS}/{model_name}'):
        mc = tf.keras.callbacks.ModelCheckpoint(f'./{INPUT_LENGTH}_{OUTPUT_STEPS}/{model_name}', monitor='val_loss', mode='min', save_best_only=True)

        model.compile(loss=tf.losses.MeanSquaredError(),           # Use the MSE error to compute the quality!!
                      #optimizer=tf.optimizers.Adam(learning_rate=0.001),
                      optimizer=tf.optimizers.RMSprop(),
                      metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredError(), tf.metrics.RootMeanSquaredError(), tfa.metrics.RSquare()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping, mc])
    else:
        print(f'{model_name} already trained')

    return history


def get_num_instances(dataset):
    instances = 0
    for data in dataset:
        x, y = data
        instances += x.shape[0]   # number of instances here
    return instances


# dataset, figures, models paths:

dataset_path = "SUSTITUIR POR EL DIRECTORIO DEL ALUMNO DONDE SE ALMACEN LOS DATOS FACILITADOS" (es una cadena)
figures_path = "SUSTITUIR POR EL DIRECTORIO DEL ALUMNO DONDE SE ALMACENEN LAS FIGURAS"
models_path = "SUSTITUIR POR EL DIRECTORIO DEL ALUMNO DONDE SE ALMACENEN LOS MODELOS GENERADOS"


# Read the data:
df = pd.read_csv(dataset_path + 't_msb1m_sites_metadata_date_meteo_cluster.csv', delimiter=';')
df.fillna(axis=1, value=0, inplace=True)

# Convert columns to its datatype:
df['01.01 UNIT site'] = df['01.01 UNIT site'].astype(int)
df['0.1 TIME date'] = pd.to_datetime(df['0.1 TIME date'])
for i in range(3, 24):
    df[df.columns[i]] = df[df.columns[i]].astype(float)



# Drop '0.1 TIME date' and 'cluster' columns as they are not useful for this problem. They are descriptive variables only.
df = df.drop('0.1 TIME date', axis=1)
df = df.drop('cluster', axis=1)

# NORMALIZACIÓN: Media 0, desviacion 1.
sc=StandardScaler()                    # Transformer is created
df_sc = sc.fit_transform(X=df)
df = pd.DataFrame(df_sc, columns = df.columns)

train_pct = 0.7
validation_pct = 0.2
test_pct = 0.1

# Split data into train, validation and test, we have to take the last N timesteps as test data.
num_samples = len(df)
train_data = df.iloc[:int(num_samples * train_pct), :]                                                  # 70% train data
val_data = df.iloc[int(num_samples * train_pct):int(num_samples * (train_pct + validation_pct)), :]     # 20% validation data
test_data = df.iloc[int(num_samples * (train_pct + validation_pct)):, :]                                # 10% test data

INPUT_LENGTH = int(args.n_input)                     # get the last 24 hours for make the prediction
INPUT_VARIABLES = train_data.shape[1]
OUTPUT_STEPS = int(args.n_output)                      # predict only the next 3 hour(s)
OUTPUT_VARIABLES = 1                  # Number of variables (timeseries) to predict.
BATCH_SIZE = int(args.batch_size)


# Split the dataset to create instances for deep learning in the following way for example:
# using data from t = [0,1,2,3,4]  ->  predict t = [5].   This is an instance. The next ones follow a sliding window approach:
# using data from t = [1,2,3,4,5]  ->  predict t = [6].
# using data from t = [2,3,4,5,6]  ->  predict t = [7].
# This is done the same way on the validation and test data.
CLASS_NAME = args.clase  #'.B03 Consumption kWh'
w1 = WindowGenerator(input_width=INPUT_LENGTH,
                     label_width=OUTPUT_STEPS,
                     shift=OUTPUT_STEPS,
                     train_df=train_data,
                     val_df=val_data,
                     test_df=test_data,
                     batch_size=BATCH_SIZE,
                     label_columns=[CLASS_NAME])
print(w1)

# Compute the number of instances on each dataset:
train_examples = get_num_instances(w1.make_dataset(w1.train_df))
val_examples = get_num_instances(w1.make_dataset(w1.val_df))
test_examples = get_num_instances(w1.make_dataset(w1.test_df))
total_examples = train_examples + val_examples + test_examples

print("Total number of examples:", total_examples)
print("Number of training examples:", train_examples)
print("Number of validation examples:", val_examples)
print("Number of test examples:", test_examples)

# w1.example is a tuple of values whose shape is x = (batch_size, 12 (INPUT_LENGTH), 41)), y = (batch_size, 1 (OUTPUT_STEPS), 1)
num_epochs = int(args.n_epochs)

# Here we store the results:
results = {}

# Define the input data for the CNN+LSTM model :
do_train = bool(args.train) #False
if do_train:
    input = tf.keras.Input(shape=(INPUT_LENGTH, INPUT_VARIABLES))

    # Train all models (The commented ones are not applicablles due to low number of timesteps...)
    methods_dict = {
        ##'ShiHaotian': TSFEDL.ShiHaotian(input_tensor=input, include_top=True),
        ##'YildirimOzal': TSFEDL.YildirimOzal(input_tensor=input, include_top=True),
        'OhShuLi': TSFEDL.OhShuLih(input_tensor=input, include_top=False),
        'KhanZulfiqar': TSFEDL.KhanZulfiqar(input_tensor=input, include_top=False),
        'ZhengZhenyu': TSFEDL.ZhengZhenyu(input_tensor=input, include_top=False),
        'WangKejun': TSFEDL.WangKejun(input_tensor=input, include_top=False),
        'KimTaeYoung': TSFEDL.KimTaeYoung(input_tensor=input, include_top=False),
        'GenMinxing': TSFEDL.GenMinxing(input_tensor=input, include_top=False),
        'FuJiangmeng': TSFEDL.FuJiangmeng(input_tensor=input, include_top=False),
        ##'HuangMeiLing': TSFEDL.HuangMeiLing(input_tensor=input, include_top=True),
        'GaoJunLi': TSFEDL.GaoJunLi(input_tensor=input, include_top=False),
        ##'WeiXiaoyan': TSFEDL.WeiXiaoyan(input_tensor=input, include_top=True),
        'KongZhengmin': TSFEDL.KongZhengmin(input_tensor=input, include_top=False),
        'CaiWenjuan': TSFEDL.CaiWenjuan(input_tensor=input, include_top=False),
        'HtetMyetLynn': TSFEDL.HtetMyetLynn(input_tensor=input, include_top=False),
        ##'ZhangJin': TSFEDL.ZhangJin(input_tensor=input, include_top=True),
        ##'YaoQihang': TSFEDL.YaoQihang(input_tensor=input, include_top=True),
        ##'YiboGao': TSFEDL.YiboGao(input_tensor=input, include_top=True, return_loss=False),
        'SharPar': TSFEDL.SharPar(input_tensor=input, include_top=False),
        'DaiXiLi': TSFEDL.DaiXiLi(input_tensor=input, include_top=False),
    }

    for name, model in methods_dict.items():
        print("Training: ", name)

        # Linear output for generating the predicted timestep of the target timeserios
        linear_model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                # tf.keras.layers.Dropout(rate=0.2),
                # tf.keras.layers.Dense(units=512, activation="relu"),
                # tf.keras.layers.Dropout(rate=0.2),
                # tf.keras.layers.Dense(units=256, activation="relu"),
                tf.keras.layers.Dense(OUTPUT_STEPS * OUTPUT_VARIABLES, kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([OUTPUT_STEPS, OUTPUT_VARIABLES])

            ]
        )

        # Conect the model output to the linear model (top module) and create the new model
        x = model.output
        out = linear_model(x)
        model2 = tf.keras.Model(inputs=input, outputs=out)
        model2.summary()

        # Train the model
        res = compile_and_fit(model2, w1, MAX_EPOCHS=num_epochs, patience=20, model_name=str(name + '.h5'))

        # guardamos el modelo entrenado en una carpeta con nombre nº de inputs y nº de outputs.
        saved_model = tf.keras.models.load_model(f'./{INPUT_LENGTH}_{OUTPUT_STEPS}/{name}.h5')

        # Ahora toca testear, como se ha guardado el mejor modelo gracias al EarlyStopping, lo cargamos y usamos los datos de test
        eval = saved_model.evaluate(w1.test, verbose=0)
        print(name, " MAE: ", eval[1])
        results[name] = eval   # [ Loss, MAE, MSE, RMSE, R2 ]

# Perform the inference of the models and save the results:
for file in glob.glob(f'{INPUT_LENGTH}_{OUTPUT_STEPS}/*.h5'):
    # load the model
    saved_model = tf.keras.models.load_model(file)
    saved_model.summary()

    # test the model
    evaluation = saved_model.evaluate(w1.test, verbose=1)
    results[file] = evaluation
    print(f'Loss: {evaluation[0]},  MAE: {evaluation[1]},  RMSE: {evaluation[2]},   R2: {evaluation[3]}')
    graph = w1.plot(saved_model, CLASS_NAME, max_subplots=6)
    graph.savefig(f'{file}.svg')



#%%
# write results to file
with open(f'{INPUT_LENGTH}_{OUTPUT_STEPS}_results_deeplearning.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in results.items():
        writer.writerow(i)

