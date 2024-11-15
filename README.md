# AIRCRAFT TURBO-FAN ENGINES’ REMAINING USEFUL LIFE PREDICTION USING LSTM NEURAL NETWORK MODEL AND DEEP PCA

## Overview  
**Abstract**  
This study presents the development of a Long Short-Term Memory (LSTM) neural network model, specifically designed to predict the Remaining Useful Life (RUL) of aircraft engines. Recognizing the unique operational characteristics of each engine, the research advocates for the creation of individualized models per engine, enhancing the precision of RUL predictions. The model was trained and validated on a comprehensive dataset comprising 16,321 data points, collected from 21 sensors across 100 different engines. A feature sensitivity analysis was conducted to pinpoint the most significant sensors that effectively quantify engine degradation, thereby streamlining the model's focus on the most impactful data.  
To further refine the model, a Deep Principal Component Analysis (Deep PCA) approach was employed, effectively simplifying the model's architecture while retaining essential features. The LSTM model was uniquely structured with alternating layers of 'tanh' and 'relu' activation functions, a configuration that significantly enhanced its performance in handling time-series data. The model's efficacy was rigorously evaluated using a combination of loss metrics, Mean Absolute Error (MAE), and a detailed analysis of residuals versus predictions plots. These evaluation metrics provided comprehensive insights into the model's accuracy and predictive capabilities. The outcome of this research offers a robust and tailored approach to predicting the RUL of aircraft engines, contributing valuable advancements in the field of predictive maintenance and operational efficiency in the aviation industry.  

**Introduction**  
In this study, each engine's operational time series commences under normal conditions, subsequently developing a fault as the series progresses. Within the training dataset, the fault's severity escalates until it culminates in system failure. Conversely, the time series in the test dataset concludes prior to the occurrence of system failure. The primary aim is to forecast the number of remaining operational cycles before failure in the test set, specifically the duration of operation post the final observed cycle.  
The dataset was divided into two distinct subsets: training and testing. The training set serves the initial analysis and the first objective of this paper, which involves conducting a feature sensitivity analysis to pinpoint those features that are critical in predicting engine failures. Building upon the insights gained from the first objective, the testing set is utilized for the second objective. This involves the development of a machine learning model, informed by the findings of the feature analysis, to accurately predict the RUL of the engines.  
Predictive analytics and time series analysis have seen the development of various methodologies to tackle forecasting and pattern recognition challenges. Recurrent Neural Networks (RNNs) are notable for processing sequential data, ideal for time-dependent datasets. Auto-regressive models like ARIMA provide a statistical approach for time series forecasting, while Multi-level Regression Models handle hierarchical data structures. Linear Models with Interactions offer insights into variable interactions, and Customized Decision-Trees and Regression Models allow for domain-specific adaptations. Anomaly Detection Models play a key role in identifying data outliers, crucial for quality control. Convolutional Neural Networks (CNNs), known for image processing, are also used in time series analysis for their pattern recognition capabilities.  
Amidst these varied methodologies, LSTM Neural Networks have emerged as a particularly potent tool, especially in the context of RUL analysis. LSTM networks, a specialized kind of RNN, are adept at capturing long-term dependencies in time series data, a critical requirement for accurately predicting the RUL of machinery and components. This capability is invaluable in industries where maintenance and operational efficiency are paramount, as it enables the prediction of the time frame within which a component or system is likely to function reliably. The application of LSTM networks in RUL analysis represents a significant advancement in predictive maintenance strategies, offering a more data-driven, precise, and cost-effective approach to managing the lifecycle of critical assets.  

**Executive summary**  
1. Data Preparation and Analysis:  
- Comprehensive preprocessing, including normalization and standardization, was applied to the dataset, derived from a variety of engine sensors, to ensure consistency across different features.  
- Non-contributory features exhibiting zero variance, such as 'Operation Condition 3', 'T2', 'P2', 'P15', 'epr', 'farB', 'Nf_dmd', and 'PCNfR_dmd', were identified and excluded.  
- A new target variable, 'remaining_cycles', was created, representing the difference between the total number of cycles and the current cycle number, to guide the model training process.  
2. Model Development and Evaluation:  
- The study employed a dual-model approach, incorporating both a Deep PCA model (autoencoder) and a Long Short-Term Memory (LSTM) neural network, each optimized for the specific characteristics of time-series engine data.  
- The Deep PCA model focused on dimensionality reduction while preserving essential features, whereas the LSTM model, with its alternating 'tanh' and 'relu' activation functions, was designed to capture complex temporal dependencies in the data.  
- Both models underwent rigorous training and validation, with the LSTM model additionally subjected to a residual analysis to evaluate its predictive accuracy.  
3. Outliers and Collinearity:  
- Outliers, particularly prevalent towards the end of the engines' operational cycles, were retained due to their significance in predicting the final cycle.  
- The model also addressed collinearity issues, with the removal of redundant features based on correlation analysis, thereby reducing model complexity without compromising data integrity.  
4. Generalizability and Application:  
- While Engine 65 was the primary focus, the methodologies and techniques developed are universally applicable across other engines. This adaptability underscores the model's potential for broader application in predictive maintenance and operational optimization in the aviation industry.  

## Data analysis  
**Significant sensors sufficient to quantify the degradation of aircraft turbofan engines (Feature sensitivity analysis)**  
<img src="https://github.com/user-attachments/assets/3c13762a-f0a6-4fda-9eeb-1051cfad1903" alt="image" width="800" height="auto">  
Certain features exhibiting no variance were identified as non-contributory to the model's performance, as illustrated in Figure 6. These features include 'Operation Condition 3', 'T2', 'P2', 'P15', 'epr', 'farB', 'Nf_dmd', and 'PCNfR_dmd'. The lack of variance in these features indicates their limited utility in enhancing the predictive capability of the model, leading to their exclusion from the analysis to streamline the model and focus on more impactful variables.  

**Data distribution**  
<img src="https://github.com/user-attachments/assets/2df94ebb-831d-423c-95a5-fe7014c30f08" alt="image" width="auto" height="auto"> 
In our methodology, features exhibiting zero variance were omitted to streamline the model. These excluded features include 'Operation Condition 3', 'T2', 'P2', 'P15', 'epr', 'farB', 'Nf_dmd', and 'PCNfR_dmd'. As depicted in Figure 1, the input data for some features follows a normal distribution, whereas others do not. Furthermore, the input data encompasses a broad range, attributable to its derivation from diverse sensor sources. Consequently, implementing normalization and standardization procedures on this dataset is deemed advantageous for enhancing model performance.  

**Collinearity**  
<img src="https://github.com/user-attachments/assets/99c3d8da-415f-4173-821a-92a369ca6fed" alt="image" width="auto" height="auto">  
Analysis reveals a strong positive correlation between the features NRc and Nc. Based on this observation, Nc was excluded from the model to reduce complexity and avoid redundancy in the dataset. This decision was made to streamline the model while retaining the essential characteristics of the data.  

**Outliers**  
  <table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2513b156-9412-44bf-9dad-5207816d4b59" width="500"/><br/>
      <b>Detection of outliers on some features</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ef6b4940-503f-4cd5-b563-7da1ae18dcb2" width="500"/><br/>
      <b>Outliers vs cycle numbers</b>
    </td>
  </tr>
</table> 
The chart on the left illustrates the presence of a considerable number of outliers within our dataset. Typically, the removal of such outliers is conducive to enhancing prediction accuracy. However, as depicted in the chart on the right, a majority of these outliers are observed towards the latter part of the engine's operational cycle. Given that our primary objective is to predict the remaining operational cycles of the engine, the retention of these outliers in our model is anticipated to contribute positively to its performance. This approach acknowledges the critical role of these data points in capturing the end-of-life behavior of the engine, which is essential for accurate predictive modeling in this context.  

**Data trends**  
<img src="https://github.com/user-attachments/assets/4e7be884-6036-4ad2-9448-fa8e090d644f" alt="image" width="auto" height="auto"> 
For identical sensors, data trends vary significantly across different engines, particularly as they approach their final operational cycle. This variation underscores the impracticality of formulating a universal model applicable to all engines. Consequently, we have opted to develop a specialized machine learning pipeline tailored to predict the remaining operational cycles for each individual engine. This approach allows for a more nuanced and accurate prediction by accounting for the unique characteristics and performance patterns of each engine.  

## Methodology  
**Data processing**  
The sensor input data in our study were subjected to normalization and standardization processes to ensure uniformity in scale across different features. This step was crucial to mitigate any potential numerical biases that might arise due to variations in the magnitude of data across different sensors.  
Additionally, a new column, termed 'remaining_cycles', was created and designated as the target variable for model training. This variable was derived by calculating the difference between the total number of cycles and the current cycle number for each engine. The primary objective of the model is to accurately predict the number of remaining operational cycles at any given point in an engine's lifecycle.  
As previously mentioned, the presence of outliers, particularly those indicative of the final cycle of engine operation, was deemed beneficial for the predictive accuracy of the model. Therefore, these outliers were retained in the datasets to enhance the model's ability to forecast the end-of-life point for each engine accurately.  

**Forecasting the number of cycles remaining to failure**  
In this study, Engine 65 served as the primary subject for constructing our machine learning model. This specific engine was selected as a representative case to demonstrate the applicability of our methodology. However, it is important to note that the developed approach is not exclusive to Engine 65. The methodology and the modeling techniques employed are designed to be universally applicable across other engines. This versatility ensures that the insights and predictive capabilities derived from the model can be effectively extended to the remaining engines in the dataset, thereby providing a comprehensive and adaptable solution for engine lifecycle analysis.  

1. Deep PCA model  
An autoencoder neural network was developed to process our dataset. The architecture of our autoencoder was designed with two initial hidden layers and a middle layer. The number of neurons in these layers, n1 and n2, was optimized using Keras Tuner, a hyperparameter tuning framework. This optimization process determined that both the initial hidden layers and the middle layer should consist of 128 neurons each.  
The autoencoder was constructed using a custom function, tailored to our dataset's characteristics. We compiled the autoencoder using the Adam optimizer and mean squared error as the loss function.  
For training and validation, the dataset was split into two subsets: a training set and a validation set, with the latter accounting for 20% of the data. This split was performed using a random seed of 42 to ensure reproducibility. The model was then trained over 50 epochs with a batch size of 256, using the training set for both input and output, and the validation set for model evaluation.  
Post training, we extracted the encoder part of the autoencoder. This encoder, a sub-model of the autoencoder, was defined to have the same input as the autoencoder but output from the 'encoder2' layer. We utilized this encoder to transform the entire training dataset, effectively reducing its dimensionality while retaining essential features.  

2. LSTM model  
<img src="https://github.com/user-attachments/assets/1d5063b1-b50b-4e00-91bb-a7ac8b2a0cd5" alt="image" width="auto" height="auto"> 
We developed a Long Short-Term Memory (LSTM) neural network model tailored for time-series data analysis. The architecture of the LSTM model, particularly the strategic use of 'tanh' and 'relu' activation functions across its layers, demonstrates a methodical approach to enhancing the model's ability to process and analyze time-series data effectively. The inclusion of multiple LSTM and dropout layers reflects a balance between model complexity and the risk of overfitting, aiming to achieve high accuracy in predictions while maintaining generalizability.  
The model's input shape is defined by (X_train.shape[1], X_train.shape[2]), which corresponds to the number of timesteps and features in the training data, respectively.  

- LSTM Layers:  
  - The model consists of multiple LSTM layers, each with 200 units. LSTM, a type of recurrent neural network, is particularly effective in capturing temporal dependencies and patterns in time-series data.
  - The activation='tanh' and activation='relu' parameters in alternating LSTM layers introduce non-linearity, allowing the model to learn complex relationships in the data. The hyperbolic tangent (tanh) activation function processes data within the range (-1, 1), while the rectified linear unit (ReLU) activation function allows for faster training and mitigates the vanishing gradient problem.  
  - The return_sequences=True parameter in all but the last LSTM layer ensures that the output of each LSTM layer is a sequence, which is necessary for stacking LSTM layers. The final LSTM layer has return_sequences=False, making its output suitable for feeding into the subsequent Dense layer.  

- Dropout Layers: Following each LSTM layer, a Dropout layer with a rate of 0.2 is applied. This means 20% of the neurons' outputs are randomly set to zero during training. Dropout is a regularization technique used to prevent overfitting by reducing the model's sensitivity to specific features of the training data.  

- Dense Layers:  
  - After the LSTM and Dropout layers, the model includes a Dense layer with 10 neurons, using the ReLU activation function. This layer serves as a fully connected layer that processes the features learned by the LSTM layers.  
  - The final layer of the model is another Dense layer with a single neuron. This layer typically acts as the output layer for the model. The absence of an activation function implies a linear activation, which is common in regression models where the output is a continuous value.  
The model is then compiled using Adam method at the learning rate of 0.001.

3. Performance  
<img src="https://github.com/user-attachments/assets/062b49b6-50eb-457b-9119-c198731c923a" alt="image" width="auto" height="auto">  
In our computational experiments, as shown in figure 8, the model demonstrated rapid convergence at timesteps 35 and 45. Notably, at those timesteps mark, the model tended to settle into a local optimum, indicating a potential limitation in exploring the solution space. Conversely, at timesteps 15 and 25, the model consistently achieved a lower Mean Absolute Error (MAE), outperforming other timestep configurations. Despite some instability in proximity to the optimal solution, the 25-timestep configuration emerged as the most effective for training the model, balancing convergence speed and predictive accuracy., achieving the MAE of 1.8432.  

<img src="https://github.com/user-attachments/assets/e1da15ba-2d28-49c8-948f-29e8200a8c0a" alt="image" width="500" height="auto">  
It is evident that the model with a timestep of 25 exhibits the lowest residuals. These residuals are uniformly distributed around the zero mark across a broad spectrum of predictions. This observation corroborates our earlier assessment, affirming that a timestep of 25 is the most advantageous selection for optimal model performance.  

## Discussion  
Throughout the development phase of our LSTM model for predicting aircraft engine RUL, various approaches were explored to ascertain the most effective strategy. Initially, the concept of a generic prediction model applicable to all engines was pursued. However, this approach yielded limited success, suggesting that the variability and unique characteristics of each engine's operational data might be too diverse to be encapsulated effectively by a single, overarching model. Similarly, attempts to create models for clusters of engines, grouped based on certain shared characteristics, also did not achieve the desired level of predictive accuracy. These outcomes indicate that the complexities inherent in aircraft engine data might necessitate a more comprehensive and sophisticated modeling approach, possibly one that can adapt more dynamically to the nuances of each engine's data.  
Looking ahead to future work, there are several avenues for enhancing the model's performance and accuracy. One such direction involves the application of advanced hyperparameter tuning techniques. Tools like GridSearchCV offer a systematic method for exploring a wide range of hyperparameters, thereby identifying the optimal configuration for the model. This process could be instrumental in fine-tuning various aspects of the LSTM model, including the number of units in the hidden layers, the choice of activation functions, and the selection of the most effective optimizer and learning rate. By exhaustively searching through the hyperparameter space, GridSearchCV can significantly contribute to improving the model's ability to predict RUL with greater precision.  
