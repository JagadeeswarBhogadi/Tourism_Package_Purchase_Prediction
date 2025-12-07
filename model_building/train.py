import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# -------------------
# MLflow Setup
# -------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment-v2")

# Ensure previous run is ended
if mlflow.active_run() is not None:
    mlflow.end_run()

# -------------------
# HuggingFace Login
# -------------------
hf_token = os.getenv("HF_TOKEN")


login(token=hf_token)
api = HfApi()
print("✅ Logged into HuggingFace successfully.\n")

# -------------------
# Load Data from Hugging Face Hub
# -------------------
repo_id = "Jagadesswar/tourism-prediction"

Xtrain_path = hf_hub_download(
    repo_id=repo_id,
    filename="Xtrain.csv",
    repo_type="dataset"
)

Xtest_path = hf_hub_download(
    repo_id=repo_id,
    filename="Xtest.csv",
    repo_type="dataset"
)

ytrain_path = hf_hub_download(
    repo_id=repo_id,
    filename="ytrain.csv",
    repo_type="dataset"
)

ytest_path = hf_hub_download(
    repo_id=repo_id,
    filename="ytest.csv",
    repo_type="dataset"
)

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()

print("✅ Datasets loaded successfully.")

categorical_cols = Xtrain.select_dtypes(include=["object"]).columns.tolist()
num_cols = Xtrain.select_dtypes(exclude=["object"]).columns.tolist()

# -------------------
# Preprocessing Pipeline
# -------------------
preprocessor = make_column_transformer(
    (StandardScaler(), num_cols),
    (OneHotEncoder(handle_unknown="ignore"), categorical_cols)
)

# Handle class imbalance
class_weight = list(ytrain).count(0) / max(list(ytrain).count(1), 1)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -------------------
# MLflow Training
# -------------------
with mlflow.start_run():
    # Grid Search
    grid = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_
    mlflow.log_params(grid.best_params_)

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score']
    })

    # -------------------
    # Save Model
    # -------------------
    save_dir = "tourism_project/model_building"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_tourism_model_v1.joblib")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)
    print(f"✅ Model saved at: {model_path}")

    # -------------------
    # Upload Model to HuggingFace
    # -------------------
    hf_repo_id = "Jagadesswar/tourism_prediction_model"  # replace YOUR_USERNAME
    try:
        api.repo_info(hf_repo_id, repo_type="model")
        print("📦 Model repository exists.")
    except RepositoryNotFoundError:
        print("📦 Creating new model repository...")
        create_repo(hf_repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=hf_repo_id,
        repo_type="model"
    )

    print("🚀 Model uploaded to HuggingFace successfully!")
