import numpy as np
import csv
import io
from google.colab import files
import pandas as pd
from sklearn.model_selection import train_test_split

file = files.upload()

df_vectorStatus = pd.read_csv(io.StringIO(file['vectorStatus.csv'].decode('utf-8')), index_col=False)
df_vectorStatusDur = pd.read_csv(io.StringIO(file['vectorStatusDur.csv'].decode('utf-8')), index_col=False)
df_TransitionRate = pd.read_csv(io.StringIO(file['TransitionRate.csv'].decode('utf-8')), index_col=False)
df_VarianceDur = pd.read_csv(io.StringIO(file['VarianceDur.csv'].decode('utf-8')), index_col=False)

# Reset indices
df_vectorStatus = df_vectorStatus.reset_index(drop=True)
df_vectorStatusDur = df_vectorStatusDur.reset_index(drop=True)
df_TransitionRate = df_TransitionRate.reset_index(drop=True)
df_VarianceDur = df_VarianceDur.reset_index(drop=True)

# Separate metadata columns
freq_metadata = df_vectorStatus[['ThreadID', 'WindowStartTime(s)']]
vector_status = df_vectorStatus.drop(columns=['ThreadID', 'WindowStartTime(s)'])
# Separate metadata columns
dur_metadata = df_vectorStatusDur[['ThreadID', 'WindowStartTime(s)']]
dur_status = df_vectorStatusDur.drop(columns=['ThreadID', 'WindowStartTime(s)'])
# Separate metadata columns
TransitionRate_metadata = df_TransitionRate[['ThreadID', 'WindowStartTime(s)']]
TransitionRate = df_TransitionRate.drop(columns=['ThreadID', 'WindowStartTime(s)'])
# Separate metadata columns
VarianceDur_metadata = df_VarianceDur[['ThreadID', 'WindowStartTime(s)']]
VarianceDur = df_VarianceDur.drop(columns=['ThreadID', 'WindowStartTime(s)'])

# Rename columns for frequency data
vector_status = vector_status.add_prefix('freq_')

# Rename columns for duration data
dur_status = dur_status.add_prefix('dur_')

# # Rename columns for frequency data
# TransitionRate = TransitionRate.add_prefix('trans_')

# # Rename columns for duration data
# VarianceDur = VarianceDur.add_prefix('var_')


# Concatenate the frequency and duration data
full_data = pd.concat([freq_metadata, vector_status,dur_metadata, dur_status, TransitionRate_metadata,TransitionRate, VarianceDur_metadata, VarianceDur], axis=1)
# full_data = pd.concat([freq_metadata, vector_status,dur_metadata, dur_status], axis=1)

# Drop duplicate 'ThreadID' and 'WindowStartTime(s)' columns
full_data = full_data.loc[:, ~full_data.columns.duplicated()]


# Define a threshold to filter data for augmentation (example: augment data where WindowStartTime(s) > a certain value)
#
filtered_df = full_data[full_data["WindowStartTime(s)"] < 1200]
print(filtered_df.shape)

# Data augmentation: Add small Gaussian noise to numeric columns for selected data
augmented_df = filtered_df.copy()
# Select all numeric columns except 'ThreadID' and 'WindowStartTime(s)'
numeric_columns = [col for col in filtered_df.columns if col not in ['ThreadID', 'WindowStartTime(s)']]

for col in numeric_columns:
    std_dev = filtered_df[col].std()

    # Skip columns with zero or NaN standard deviation
    if pd.isna(std_dev) or std_dev == 0:
        continue

    # Add Gaussian noise
    noise = np.random.normal(0, 0.01 * std_dev, size=filtered_df.shape[0])
    augmented_df[col] = filtered_df[col] + noise

    # Replace NaN and Inf values with 0
    augmented_df[col] = augmented_df[col].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).astype(int)




# Assign 6-digit ThreadID starting from 100000 for augmented data
augmented_df["ThreadID"] = np.random.randint(100000, 999999, size=augmented_df.shape[0])


# Combine original and augmented data without mixing them
augmented_df["Augmented"] = True  # Flag for augmented data
full_data["Augmented"] = False           # Flag for original data
combined_df = pd.concat([full_data, augmented_df])

augmentation_factor = 3  # This will triple the number of augmented rows

# Repeat the augmentation process
augmented_repeated_df = pd.concat([augmented_df.copy() for _ in range(augmentation_factor)], ignore_index=True)

# Assign new unique 6-digit ThreadIDs for each repetition to keep them distinct
augmented_repeated_df["ThreadID"] = np.random.randint(100000, 999999, size=augmented_repeated_df.shape[0])

augmented_repeated_df["Augmented"] = True  # Flag for augmented data
combined_df = pd.concat([full_data, augmented_repeated_df])

print(combined_df.shape)

original_full_data= combined_df.copy()
# Split into 80% training and 20% testing
train_data, test_data = train_test_split(combined_df, test_size=0.2, random_state=42)

original_test_data= test_data.copy()



import tensorflow

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from google.colab import files
import csv
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import io

# Normalize the data
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)


X_train = train_data_scaled.reshape(train_data_scaled.shape[0], 1, train_data_scaled.shape[1])
print("Training data shape:", X_train.shape)
X_test = test_data_scaled.reshape(test_data_scaled.shape[0], 1, test_data_scaled.shape[1])
print("Test data shape:", X_test.shape)

# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=True)(L1)
    L3 = LSTM(16, activation='relu', return_sequences=True)(L2)
    output = TimeDistributed(Dense(X.shape[2]))(L3)
    model = Model(inputs=inputs, outputs=output)
    return model

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import time

nb_epochs = 50
batch_size = 100


X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)



t_exe= []
a, r, p, f1= [], [], [], []
for i in range(10):
  # create the autoencoder model
  start_time = time.time()
  model = autoencoder_model(X_train)
  #model.compile(optimizer='adam', loss='mae', metrics = ['accuracy'])
  model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['accuracy'])
  # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])



  history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.05).history
  X_pred = model.predict(X_test)
  X_pred_train = model.predict(X_train)

  test_data = X_test.flatten()
  pred_data = X_pred.flatten()

  print(X_test.shape, X_pred.shape)

  # Use a threshold instead of rounding if needed
  threshold = 0.05
  pred_data_binary = (pred_data > threshold).astype(int)
  test_data_binary = (test_data > threshold).astype(int)

  # Calculate metrics
  accuracy = accuracy_score(test_data_binary, pred_data_binary)
  precision = precision_score(test_data_binary, pred_data_binary, average='macro', zero_division=1)
  recall = recall_score(test_data_binary, pred_data_binary, average='macro', zero_division=1)
  f1 = f1_score(test_data_binary, pred_data_binary, average='macro')

  print('Accuracy:', accuracy)
  print('Precision:', precision)
  print('Recall:', recall)
  print('F1 score:', f1)

  t= time.time() - start_time
  t_exe.append(t)

  a.append(accuracy)
  r.append(recall)
  p.append(precision)
  #f1.append(f1_score)

  # Evaluate on test data
  reconstructions = model.predict(X_test)
  reconstruction_errors = np.mean(np.square(reconstructions - X_test), axis=(1, 2))

  # Detect anomalies
  threshold = np.percentile(reconstruction_errors, 95)  # Set threshold
  anomalies = reconstruction_errors > threshold


  print(i)

acc= np.max(a)
rec= np.max(r)
perc= np.max(p)
#f_score= np.max(f1)
avg_time= np.average(t_exe)


print("--- Average execution time is: %s seconds ---" % avg_time)

print("Accuray: ", acc)
print("Recall: ", rec)
print("Precision: ", perc)
#print("F1 score: ", f_score)


print(f"Number of anomalies: {np.sum(anomalies)}")
print(f"Anomaly indices: {np.where(anomalies)[0]}")

# Step 1: Extract anomalies from the original data
anomaly_indices = np.where(anomalies)[0]
original_anomalies = original_test_data.iloc[anomaly_indices]

# Step 2: Force unique index to avoid reindexing issues
original_anomalies = original_anomalies.reset_index(drop=True)
original_anomalies.index = range(len(original_anomalies))  # Ensure unique, continuous indexing

# Step 3: Check if the 'Augmented' column exists
if "Augmented" in original_anomalies.columns:
    # Step 4: Filter out augmented data
    filtered_anomalies = original_anomalies.loc[original_anomalies["Augmented"] == False]

    # Step 5: Display the non-augmented anomalies
    print("Original Non-Augmented Anomalous Data Points:\n")
    print(filtered_anomalies.head(5))
else:
    print("Error: 'Augmented' column is missing in the dataset.")

# Save to CSV
filtered_anomalies.to_csv(r'anomalies.csv', index=False)
print(len(anomalies))

import matplotlib.pyplot as plt
import seaborn as sns

# Plot reconstruction error distribution
plt.figure(figsize=(10, 6))
sns.histplot(reconstruction_errors, bins=50, kde=True, color='blue', label='Normal')
sns.histplot(reconstruction_errors[anomalies], bins=50, kde=True, color='red', label='Anomalies')

plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))

# Plot reconstruction errors over actual time
plt.scatter(original_test_data['WindowStartTime(s)'], reconstruction_errors,
            label='Reconstruction Error', color='blue', alpha=0.6, s=10)

# Highlight anomalies with larger red points
plt.scatter(original_test_data['WindowStartTime(s)'][anomalies],
            reconstruction_errors[anomalies],
            color='red', label='Anomalies', s=30)

# Plot the threshold line
plt.axhline(y=threshold, color='black', linestyle='--', label='Threshold')

# Axis labels and title
plt.xlabel("Window Start Time (s)")
plt.ylabel("Reconstruction Error")
plt.title("Anomalies Over Time")
plt.legend()
plt.show()

from sklearn.decomposition import PCA

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_test.reshape(X_test.shape[0], -1))

plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[~anomalies, 0], reduced_data[~anomalies, 1],
            label='Normal', alpha=0.6, edgecolor='k')
plt.scatter(reduced_data[anomalies, 0], reduced_data[anomalies, 1],
            label='Anomaly', alpha=0.9, edgecolor='red', color='red')
plt.title("PCA Projection of Test Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Plot reconstruction error distribution
plt.figure(figsize=(10, 6))
sns.histplot(reconstruction_errors, bins=50, kde=True, color='blue')

plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming X_test is your test dataset and 'anomalies' is a boolean array of detected anomalies

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_test.reshape(X_test.shape[0], -1))  # Flatten if necessary

# Plotting the t-SNE projection
plt.figure(figsize=(10, 6))

# Normal data points
plt.scatter(X_tsne[~anomalies, 0], X_tsne[~anomalies, 1],
            label="Normal", alpha=0.6, c='blue', s=10)

# Anomalies
plt.scatter(X_tsne[anomalies, 0], X_tsne[anomalies, 1],
            label="Anomaly", alpha=0.8, c='red', s=20)

plt.title("t-SNE Projection of Test Data")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.show()

pip install umap-learn

import umap
import matplotlib.pyplot as plt

# Apply UMAP for dimensionality reduction
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_test.reshape(X_test.shape[0], -1))

# Plotting the UMAP projection
plt.figure(figsize=(10, 6))

# Normal data points
plt.scatter(X_umap[~anomalies, 0], X_umap[~anomalies, 1],
            label="Normal", alpha=0.6, c='blue', s=10)

# Anomalies
plt.scatter(X_umap[anomalies, 0], X_umap[anomalies, 1],
            label="Anomaly", alpha=0.8, c='red', s=20)

plt.title("UMAP Projection of Test Data")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.show()