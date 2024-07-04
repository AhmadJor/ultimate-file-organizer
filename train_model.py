import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Load configuration from JSON file
def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Configuration file not found.")
        raise
    except json.JSONDecodeError:
        print("Error decoding JSON configuration file.")
        raise

config = load_config()

# Load training data
def load_data(data_dir):
    data = []
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for filename in os.listdir(category_dir):
                file_path = os.path.join(category_dir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        data.append((content, category))
    return pd.DataFrame(data, columns=['text', 'category'])

# Train and save the model
def train_and_save_model(data_dir, model_path):
    data = load_data(data_dir)
    if data.empty:
        print("No data found for training.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['category'], test_size=0.2, random_state=42)
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Update the model with new data
def update_model(data_dir, model_path):
    data = load_data(data_dir)
    if data.empty:
        print("No new data found for updating the model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['category'], test_size=0.2, random_state=42)
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        model.named_steps['multinomialnb'].partial_fit(X_train, y_train, classes=np.unique(data['category']))
    else:
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Updated Accuracy:", accuracy_score(y_test, y_pred))
    print("Updated Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"Updated model saved to {model_path}")

if __name__ == '__main__':
    data_dir = config['training_data_dir']
    model_path = 'file_classifier_model.pkl'

    # Train and save the initial model
    train_and_save_model(data_dir, model_path)

    # Update the model with new data (if any)
    new_data_dir = config.get('new_data_dir')
    if new_data_dir and os.path.exists(new_data_dir):
        update_model(new_data_dir, model_path)
    else:
        print(f"New data directory '{new_data_dir}' does not exist or is empty. Skipping model update.")
