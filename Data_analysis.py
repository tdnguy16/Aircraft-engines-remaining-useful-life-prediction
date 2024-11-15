import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'train_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset, the data types of each column, and some summary statistics
data_head = data.head()
data_info = data.info()
data_description = data.describe()

data_head, data_info, data_description

data = data.drop(
    ['Operation Condition 3', 'T2', 'P2', 'P15', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd'],
    axis=1)

########################################################################
# Selecting a subset of engines for trend analysis
selected_engines = data['Engine id'].unique()[:5]  # Selecting the first 5 engines
selected_parameters = ['T30', 'T50', 'P30', 'Nf', 'Nc']

# Plotting trends for these parameters for each of the selected engines
plt.figure(figsize=(15, 10))

for i, param in enumerate(selected_parameters, 1):
    plt.subplot(len(selected_parameters), 1, i)
    for engine in selected_engines:
        subset = data[data['Engine id'] == engine]
        sns.lineplot(x=subset['Cycle number'], y=subset[param], label=f'Engine {engine}')
    plt.ylabel(param)
    plt.xlabel('Cycle number')
    if i == 1:
        plt.title('Parameter Trends by Engine')


########################################################################
# Plotting outliers for each selected parameter against the cycle number
# Number of rows and columns for the subplot grid
n_rows = 4
n_cols = 4

data_dist = data.drop(['Cycle number', 'Engine id'], axis=1)

# Creating a large combined chart with histograms for each feature
plt.figure(figsize=(20, 24))

for i, column in enumerate(data_dist.columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(data_dist[column], kde=False, bins=30)
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.suptitle('Histograms for All Features in the Dataset', y=1.02, fontsize=16)


########################################################################
# Correlation Analysis
correlation_matrix = data.drop(['Engine id'], axis=1).corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Selected Parameters')


########################################################################
# Anomaly Detection using Boxplots
plt.figure(figsize=(15, 10))

for i, param in enumerate(selected_parameters, 1):
    plt.subplot(len(selected_parameters), 1, i)
    sns.boxplot(x=data[param])
    plt.xlabel(param)
    if i == 1:
        plt.title('Boxplots for Anomaly Detection')

########################################################################
# Plotting outliers for each selected parameter against the cycle number
def identify_outliers(df, columns):
    """
    Identify outliers in the specified columns of a dataframe using the IQR method.
    Returns a dataframe with an additional column 'Outlier' indicating if the row is an outlier.
    """
    outlier_indices = []

    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        outlier_indices.extend(column_outliers)

    df['Outlier'] = 0
    df.loc[outlier_indices, 'Outlier'] = 1
    return df

# Re-identifying outliers in the dataset with the fixed function
data_with_outliers = identify_outliers(data.copy(), selected_parameters)

# Splitting the data into outliers and non-outliers for the fixed dataset
outliers = data_with_outliers[data_with_outliers['Outlier'] == 1]
non_outliers = data_with_outliers[data_with_outliers['Outlier'] == 0]

# Re-comparing the distribution of cycle numbers for outliers and non-outliers
outlier_cycle_dist = outliers['Cycle number']
non_outlier_cycle_dist = non_outliers['Cycle number']

outlier_cycle_dist.describe(), non_outlier_cycle_dist.describe()


plt.figure(figsize=(15, 10))

for i, param in enumerate(selected_parameters, 1):
    plt.subplot(len(selected_parameters), 1, i)
    sns.scatterplot(x='Cycle number', y=param, data=outliers, color='red', label='Outliers')
    sns.scatterplot(x='Cycle number', y=param, data=non_outliers, color='blue', label='Non-Outliers', alpha=0.3)
    plt.ylabel(param)
    plt.xlabel('Cycle number')
    if i == 1:
        plt.title('Outliers vs. Non-Outliers for Selected Parameters Over Cycle Number')





plt.tight_layout()
plt.show()
