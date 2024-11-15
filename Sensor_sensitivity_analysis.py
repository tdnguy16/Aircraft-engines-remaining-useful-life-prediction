import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

def sensitivity_analysis(model, data, feature_names, epsilon=0.01):
    sensitivities = []
    base_prediction = model.predict(data)

    for i in range(data.shape[1]):
        # Create a copy of the data and modify the ith feature
        perturbed_data = np.copy(data)
        perturbed_data[:, i] += epsilon

        # Get the model's prediction on the perturbed data
        perturbed_prediction = model.predict(perturbed_data)

        # Calculate the sensitivity for this feature
        sensitivity = np.mean(np.abs(perturbed_prediction - base_prediction))
        sensitivities.append(sensitivity)

    return dict(zip(feature_names, sensitivities))
########################################################################
### data processing
##  Importing data
# Read the CSV file using Pandas
df_train = pd.read_csv('train_data.csv')

#TODO: clustering engines using sensors trendline slopes and build unique models for each cluster give better results.

# ##  Clustering engines
# # cluster 1
# engine_ids_to_filter_train = [1, 6, 7, 8, 12, 15, 16, 19, 22, 23, 24, 29, 30, 31, 33, 36, 37, 39, 42, 43, 46]
# # cluster 2
# # engine_ids_to_filter_train = [2, 5, 10, 20, 28, 32, 35, 44, 45, 47, 49]
#
# # Filter X_train for rows where 'Engine id' is in your list of IDs
# df_train = df_train[df_train['Engine id'].isin(engine_ids_to_filter_train)]

## Processing Training data
# Create a dictionary to hold your DataFrames
dfs_train = {}

# Loop through each unique value in the 'Engine id' column
for engine_id in df_train['Engine id'].unique():
    # Create a new DataFrame for each unique engine id
    dfs_train[engine_id] = df_train[df_train['Engine id'] == engine_id]

# Create 'remaining_cycles' col
for engine_id, df_train in dfs_train.items():
    # Calculate the maximum 'Cycle number' for the current engine
    max_cycle = df_train['Cycle number'].max()

    # Create the 'remaining_cycles' column
    df_train['remaining_cycles'] = max_cycle - df_train['Cycle number']

    # Drop cols
    df_train = df_train.drop(['Cycle number', 'Engine id'], axis=1)
    # df_train = df_train.drop(['Cycle number', 'Engine id', 'Operation Condition 3', 'T2', 'P2', 'P15', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd'], axis=1)

    # Update the DataFrame in the dictionary
    dfs_train[engine_id] = df_train

# Creating new dictionaries to hold the separated DataFrames
X_dfs_train = {}
Y_dfs_train = {}

for engine_id, df_train in dfs_train.items():
    # Y DataFrame with only the 'remaining_cycles' column
    Y_dfs_train[engine_id] = df_train[['remaining_cycles']]

    # X DataFrame with all columns except 'remaining_cycles'
    X_dfs_train[engine_id] = df_train.drop('remaining_cycles', axis=1)

# Combine data for each engine into 1 big training set
# Concatenate all DataFrames in X_dfs_train
X_train = pd.concat(X_dfs_train.values(), ignore_index=True)

# Concatenate all DataFrames in Y_dfs_train
Y_train = pd.concat(Y_dfs_train.values(), ignore_index=True)

# Inspect collinearity
# plot_correlation_heatmap(X_train)

feature_names = X_train.columns  # Replace with your actual feature names

# Filling missing values
fill_missing_values(X_train, Y_train)

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
n2 = 64  # Number of neurons in the middle layer

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
X_train_PCA = encoder.predict(X_train)


# Perform sensitivity analysis
sensitivities = sensitivity_analysis(autoencoder, X_train, feature_names)

# Plotting the sensitivities
plt.figure(figsize=(10, 6))
plt.bar(range(len(sensitivities)), list(sensitivities.values()))
plt.xticks(range(len(sensitivities)), list(sensitivities.keys()), rotation='vertical')
plt.xlabel('Feature Name')
plt.ylabel('Sensitivity')
plt.title('Feature Sensitivity Analysis')
plt.tight_layout()  # Adjust layout to fit feature names
plt.show()