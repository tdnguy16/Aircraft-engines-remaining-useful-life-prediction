import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Nadam, Ftrl


def normalize_data(X):
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def standardize_data(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized

# Scaling: Normalize or standardize your features, as LSTMs are sensitive to the scale of input data.
# Sequence Creation: Transform the data into sequences that the LSTM can process.
# Each input sequence should contain a series of data points leading up to a target 'remaining_cycles' value.
# Convert to sequences
def create_sequences(X, Y, time_steps):
    Xs, Ys = [], []

    # Check if length of Y is sufficient
    if len(Y) < len(X) - time_steps + 1:
        raise ValueError("Length of Y is insufficient. Ensure len(Y) >= len(X) - time_steps + 1.")

    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        # Check if Y is a pandas Series or DataFrame
        if isinstance(Y, (pd.Series, pd.DataFrame)):
            Ys.append(Y.iloc[i + time_steps])
        else:  # Assuming Y is a NumPy array
            Ys.append(Y[i + time_steps])
    return np.array(Xs), np.array(Ys)

#TODO: more hidden layers make improve the model accuracy but in turn slowing down the training process, the same for number of nodes at each layers.
#TODO: the combination of 'relu' and 'tanh' layers appears to give better performance
# Function to train and evaluate model
def train_and_evaluate_model(X, Y, time_steps, epochs, batch_size, optimizer):
    X_seq, Y_seq = create_sequences(X, Y, time_steps)
    X_train, X_val, Y_train, Y_val = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(200, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(200, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(200, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(200, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(200, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(200, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(200, activation='tanh', return_sequences=False),
        Dropout(0.2),
        Dense(10, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    model.evaluate(X_val, Y_val, verbose=1)

    # Generate predictions
    predictions = model.predict(X_val).flatten()  # Flatten the predictions to 1D

    # Store predictions and Y_val in the dictionaries
    predictions_dict[time_steps] = predictions
    y_val_dict[time_steps] = Y_val.flatten()  # Flatten Y_val if it's not already 1D

    return history

# Function to find missing values and forward fill
def fill_missing_values(X, Y):
    # Convert to pandas DataFrame if X and Y are not already DataFrames
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(Y, pd.DataFrame) and not isinstance(Y, pd.Series):
        Y = pd.DataFrame(Y)

    # Check and fill missing values in X
    if X.isnull().values.any():
        print("Missing values found in X. Applying forward fill...")
        X.fillna(method='ffill', inplace=True)

    # Check and fill missing values in Y
    if Y.isnull().values.any():
        print("Missing values found in Y. Applying forward fill...")
        Y.fillna(method='ffill', inplace=True)

    return X, Y

# Plot heatmap of correlation
def plot_correlation_heatmap(df):
    # Calculate the correlation matrix
    corr = df.corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap with the mask
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                vmax=1, vmin=-1, center=0, square=True, linewidths=.5)

    # Adjust the layout
    plt.tight_layout()
    # plt.show()

def create_autoencoder(input_features, n1, n2):
    """
    Creates an autoencoder model.

    :param input_features: Number of features in the input data.
    :param n1: Number of neurons in the first and second hidden layers (encoder and decoder layers).
    :param n2: Number of neurons in the middle layer (encoded representation).

    :return: A Keras model representing the autoencoder.
    """
    # Inputs
    x_tf = tf.keras.Input(shape=(input_features,))

    # Encoder
    h1_enc = tf.keras.layers.Dense(n1, activation='relu', name='encoder', use_bias=True, kernel_regularizer=tf.keras.regularizers.L1(0.0001))(x_tf)
    h1_enc = tf.keras.layers.Dropout(0.05)(h1_enc)

    # Encodings: Non-linear Principal Components
    pc = tf.keras.layers.Dense(n2, activation='relu', name='encoder2', use_bias=True, kernel_regularizer=tf.keras.regularizers.L1(0.0001))(h1_enc)
    pc = tf.keras.layers.Dropout(0.05)(pc)

    # Decoder to reconstruct
    h1_dec = tf.keras.layers.Dense(n1, activation='relu', name='decoder1', use_bias=True, kernel_regularizer=tf.keras.regularizers.L1(0.0001))(pc)
    x_recon = tf.keras.layers.Dense(input_features, activation='linear', name='decoder2', use_bias=True, kernel_regularizer=tf.keras.regularizers.L1(0.0001))(h1_dec)

    # Create the model
    autoencoder = tf.keras.Model(inputs=x_tf, outputs=x_recon)

    return autoencoder

########################################################################
### Testing data processing
##  Importing data
# Read the CSV file using Pandas
df_test_X = pd.read_csv('test_data.csv')
df_test_Y = pd.read_csv('RUL_forecast_length.csv')

# Merging the datasets on 'Engine id'
df_test = pd.merge(df_test_X, df_test_Y, on='Engine id', how='left')

df_test = df_test[df_test['Engine id'] == 65]


# # cluster 1
# engine_ids_to_filter_test = [57, 58, 59, 61, 62, 64, 65, 66, 67, 69, 72, 73, 74, 75, 76, 77, 78, 79, 81, 83, 84, 85, 86, 87, 89, 90, 91, 93, 95, 96, 98, 99, 100]
# # cluster 2
# # engine_ids_to_filter_test = [51, 52, 53, 54, 55, 56, 60, 63, 68, 70, 71, 80, 82, 88, 92, 94, 97]
#
# # Filter X_train for rows where 'Engine id' is in your list of IDs
# df_test = df_test[df_test['Engine id'].isin(engine_ids_to_filter_test)]

## Processing Training data
# Create a dictionary to hold your DataFrames
dfs_test = {}

# Loop through each unique value in the 'Engine id' column
for engine_id in df_test['Engine id'].unique():
    # Create a new DataFrame for each unique engine id
    dfs_test[engine_id] = df_test[df_test['Engine id'] == engine_id]

# Create 'remaining_cycles' col
for engine_id, df_test in dfs_test.items():
    # Calculate the maximum 'Cycle number' for the current engine
    max_cycle = df_test['Cycle number'].max() + df_test['Forecast Length'].max()

    # Create the 'remaining_cycles' column
    df_test['remaining_cycles'] = max_cycle - df_test['Cycle number']

    # Drop cols
    # df_train = df_train.drop(['Cycle number', 'Engine id'], axis=1)
    df_test = df_test.drop(['Cycle number', 'Engine id', 'Operation Condition 3', 'T2', 'P2', 'P15', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd', 'Nc'], axis=1)

    # Update the DataFrame in the dictionary
    dfs_test[engine_id] = df_test

# Creating new dictionaries to hold the separated DataFrames
X_dfs_test = {}
Y_dfs_test = {}

for engine_id, df_test in dfs_test.items():
    # Y DataFrame with only the 'remaining_cycles' column
    Y_dfs_test[engine_id] = df_test[['remaining_cycles']]

    # X DataFrame with all columns except 'remaining_cycles'
    X_dfs_test[engine_id] = df_test.drop(['remaining_cycles', 'Forecast Length'], axis=1)

# Combine data for each engine into 1 big training set
# Concatenate all DataFrames in X_dfs_train
X_train = pd.concat(X_dfs_test.values(), ignore_index=True)

# Concatenate all DataFrames in Y_dfs_train
Y_train = pd.concat(Y_dfs_test.values(), ignore_index=True)

# Normalization
X_train = normalize_data(X_train)
# Standardization
X_train = standardize_data(X_train)

########################################################################
### Deep PCA
## Training
#TODO: n1 and n2 can be further tuned to give better performance
# Create the Autoencoder Model
input_features_train = X_train.shape[1]  # Number of features in X_train
n1 = 128  # Number of neurons in the first and second hidden layers
n2 = 128  # Number of neurons in the middle layer

autoencoder = create_autoencoder(input_features_train, n1, n2)

# Compile the Model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Split X_train into training and validation sets
X_train_auto, X_val_auto = train_test_split(X_train, test_size=0.2, random_state=42)

# Train the Model
history = autoencoder.fit(X_train_auto, X_train_auto, epochs=50, batch_size=256, validation_data=(X_val_auto, X_val_auto))

# Extract the encoder part of the autoencoder
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder2').output)

# Transform the training data
X_train = encoder.predict(X_train)


########################################################################
### Modeling
#TODO: higher time steps tend to give better prediction results, even though it requires more computational power.
time_steps_list = [5, 15, 25, 35, 45]
# time_steps_list = [25]
histories = {}
predictions_dict = {}
y_val_dict = {}

#TODO: higher learning rate can speed up the training process but chances are the model will stuck a local optimas. Reducing the learning rate helps the model learn better but a lower rate.
# Define optimizer

#TODO inrease batch size to capture the randomness of datset and make the model more robust/stable.
# Train models with different time steps and store histories
for time_steps in time_steps_list:
    history_train = train_and_evaluate_model(X_train, Y_train, time_steps=time_steps, epochs=1000, batch_size=200, optimizer = Adam(learning_rate=0.001))
    histories[time_steps] = history_train

########################################################################
### Plotting
plt.figure(figsize=(12, 6))

# Plot for Loss
plt.subplot(1, 2, 1)
for time_steps, history in histories.items():
    # plt.plot(history.history['loss'], label=f'Train Loss (TS={time_steps})')
    plt.plot(history.history['val_loss'], label=f'Val Loss (TS={time_steps})')
plt.title('Model Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Plot for MAE
plt.subplot(1, 2, 2)
for time_steps, history in histories.items():
    # plt.plot(history.history['mae'], label=f'Train MAE (TS={time_steps})')
    plt.plot(history.history['val_mae'], label=f'Val MAE (TS={time_steps})')
plt.title(f'Model MAE vs Epochs')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()



# Create a single plot for all timesteps
plt.figure(figsize=(10, 8))

for timestep_key in predictions_dict.keys():
    # Calculate residuals
    residuals = y_val_dict[timestep_key] - predictions_dict[timestep_key]

    # Create a scatter plot for residuals vs predictions
    plt.scatter(predictions_dict[timestep_key], residuals, alpha=0.5, label=f'Timestep {timestep_key}')

plt.title('Residuals vs Predictions for All Timesteps')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#TODO: sensitivity to time_steps_ higher time step is better, currently at 12, higher leads to longer training time
#TODO: choose optimizer, Nadam is the best so far with very stable mae
#TODO: try GridSeachCV to pick the best parameters combination
#TODO: use customized/manual PCA
#TODO: consider regulariztion