# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
# Define constants for the dataset and output paths
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Initialize API client
api = HfApi()
print("✅ Logged into HuggingFace successfully.\n")
data = pd.read_csv("hf://datasets/Jagadesswar/tourism-prediction/tourism.csv")
df = data.copy()
df.head()

df.drop(columns=["Unnamed: 0","CustomerID"], inplace=True)

categorical_cols = []
for i in df.columns:
    if df[i].dtype == "object":
        categorical_cols.append(i)


# Encoding the categorical 'Type' column
#from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in categorical_cols:
    df[i] = label_encoder.fit_transform(df[i])

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Jagadesswar/tourism-prediction",
        repo_type="dataset",
    )
Xtrain.head()
