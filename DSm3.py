import pandas as pd
import zipfile
import urllib.request
import os

# Define the URL and file paths
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
zip_path = 'bank-additional.zip'
csv_filename = 'bank-additional/bank-additional-full.csv'  # Adjust path for the file inside the zip

# Download the dataset
print("Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)
print("Download completed.")

# Extract the zip file and list its contents
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()
    extracted_files = zip_ref.namelist()
    print(f"Extracted files: {extracted_files}")

# Ensure the correct path to the CSV file
if csv_filename not in extracted_files:
    raise FileNotFoundError(f"Expected file {csv_filename} not found in the extracted files.")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(csv_filename, sep=';')
print("Dataset loaded successfully.")

# Display the first few rows to confirm loading
print(df.head())

# Rest of the preprocessing and model training code
# Preprocess the data
df = pd.get_dummies(df, drop_first=True)
X = df.drop('y_yes', axis=1)
y = df['y_yes']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Visualize the decision tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.show()

# Clean up the downloaded zip file and extracted CSV
os.remove(zip_path)
os.remove(csv_filename)