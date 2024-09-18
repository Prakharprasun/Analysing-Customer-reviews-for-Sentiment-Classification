#By: Prakhar Prasun

from google.colab import drive
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.log_reg = LogisticRegression(max_iter=1000)
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_data(self):
        #loading data using colab
        drive.mount('/content/drive')
        self.data = pd.read_csv(self.data_path)
        self.data["sentiment"] = self.data["sentiment"].replace({'negative': 0, 'positive': 1})

    def preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    def preprocess_data(self):
        self.data['cleaned_review'] = self.data['review'].apply(self.preprocess_text)

    def vectorize_data(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['cleaned_review'])

    def split_data(self):
        X = self.tfidf_matrix
        y = self.data['sentiment']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self):
        self.log_reg.fit(self.X_train, self.y_train)
        self.random_forest.fit(self.X_train, self.y_train)

    def evaluate_model(self, true, pred):
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)
        return accuracy, precision, recall, f1

    def evaluate_models(self):
        log_reg_preds = self.log_reg.predict(self.X_test)
        rf_preds = self.random_forest.predict(self.X_test)

        log_reg_metrics = self.evaluate_model(self.y_test, log_reg_preds)
        rf_metrics = self.evaluate_model(self.y_test, rf_preds)

        print("Logistic Regression Performance:")
        print(f"Accuracy: {log_reg_metrics[0]:.4f}, Precision: {log_reg_metrics[1]:.4f}, Recall: {log_reg_metrics[2]:.4f}, F1-Score: {log_reg_metrics[3]:.4f}\n")

        print("Random Forest Performance:")
        print(f"Accuracy: {rf_metrics[0]:.4f}, Precision: {rf_metrics[1]:.4f}, Recall: {rf_metrics[2]:.4f}, F1-Score: {rf_metrics[3]:.4f}\n")

        if log_reg_metrics[3] > rf_metrics[3]:
            print("Chosen Model: Logistic Regression")
        else:
            print("Chosen Model: Random Forest")

# Using colab
data_path = '/content/drive/MyDrive/ML Datasets for induction tasks/Reviews.csv'
analysis = SentimentAnalysis(data_path)
analysis.load_data()
analysis.preprocess_data()
analysis.vectorize_data()
analysis.split_data()
analysis.train_models()
analysis.evaluate_models()
