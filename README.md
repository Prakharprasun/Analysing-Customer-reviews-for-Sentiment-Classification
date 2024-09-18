# Sentiment Analysis Using Logistic Regression and Random Forest
This project performs sentiment analysis on product reviews using Natural Language Processing (NLP) techniques and machine learning models.
# Overview
This project focuses on sentiment analysis of textual data using two supervised machine learning models: Logistic Regression and Random Forest Classifier. The dataset comprises product reviews, labeled as either positive or negative sentiments. The goal is to classify new reviews into these sentiment categories based on natural language processing (NLP) techniques such as lemmatization and TF-IDF vectorization.

Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language. 

NLP enables computers and digital devices to recognize, understand and generate text and speech by combining computational linguistics—the rule-based modeling of human language—together with statistical modeling, machine learning (ML) and deep learning. 

## Dependencies
The project requires the following Python libraries:

pandas: For data handling and manipulation.

re: For regular expression-based text cleaning.

nltk: For natural language processing (tokenization, stopword removal, lemmatization).

sklearn: For machine learning model development, vectorization, and performance evaluation.

google.colab: For accessing files stored in Google Drive (specific to Colab usage).

## Install the required packages using pip:
pip install pandas scikit-learn nltk
## Ensure that the following NLTK resources are downloaded before running the program:
nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

## Key Features
### Text Preprocessing:
Case normalization, removal of punctuation, tokenization, stopword removal, and lemmatization are performed to clean the review text.

Words like "Happy" and "happy" are semantically identical but would be treated as different tokens if case normalization isn't applied. Lowercasing ensures that these words are treated as the same entity. Most NLP models do not distinguish between cases, so standardizing words to lowercase reduces redundancy and helps avoid unnecessary distinctions between tokens that should be considered equivalent. By normalizing the case, you reduce the number of unique tokens in the dataset, making it easier for models to learn from a smaller and more cohesive vocabulary.

Punctuation marks often don’t contribute meaningful information to the sentiment of the text. They can add noise and increase the dimensionality of the data without improving model performance. Punctuation can break up words in a way that confuses tokenizers (e.g., "happy!" vs. "happy"). Removing punctuation standardizes text input and improves tokenization accuracy. Most models, including traditional methods like TF-IDF or bag-of-words, treat words as independent features. Punctuation marks would otherwise be treated as features themselves, which can distort the model’s understanding of the actual content of the text.

Tokenizers divide strings into lists of substrings. For example, tokenizers can be used to find the words and punctuation in a string.

Stop words like ‘the’, ‘and’, and ‘I’, although common, don’t usually provide meaningful information about a document’s specific topic. By eliminating these words from a corpus, we can more easily identify unique and relevant terms.

Stemmers eliminate word suffixes by running input word tokens against a pre-defined list of common suffixes. The stemmer then removes any found suffix character strings from the word, should the latter not defy any rules or conditions attached to that suffix.

The practical distinction between stemming and lemmatization is that, where stemming merely removes common suffixes from the end of word tokens, lemmatization ensures the output word is an existing normalized form of the word (for example, lemma) that can be found in the dictionary.

### TF-IDF Vectorization:
Reviews are converted into numerical form using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, capturing both the importance of a word in the review and its rarity across all reviews. Bi-grams and uni-grams are used with a maximum feature count of 1000.

Term Frequency: In document d, the frequency represents the number of instances of a given word t. Therefore, we can see that it becomes more relevant when a word appears in the text, which is rational. Since the ordering of terms is not significant, we can use a vector to describe the text in the bag of term models. For each specific term in the paper, there is an entry with the value being the term frequency.

Document Frequency: This tests the meaning of the text, which is very similar to TF, in the whole corpus collection. The only difference is that in document d, TF is the frequency counter for a term t, while df is the number of occurrences in the document set N of the term t. In other words, the number of papers in which the word is present is DF.
Inverse Document Frequency: Mainly, it tests how relevant the word is. The key aim of the search is to locate the appropriate records that fit the demand. Since tf considers all terms equally significant, it is therefore not only possible to use the term frequencies to measure the weight of the term in the paper. First, find the document frequency of a term t by counting the number of documents containing the term.

The IDF of the word is the number of documents in the corpus separated by the frequency of the text.

Tf-idf is one of the best metrics to determine how significant a term is to a text in a series or a corpus. tf-idf is a weighting system that assigns a weight to each word in a document based on its term frequency (tf) and the reciprocal document frequency (tf) (idf). The words with higher scores of weight are deemed to be more significant.
## Machine Learning Models:
Logistic Regression: A linear model for binary classification. The model predicts sentiment based on a probabilistic framework. Similar to linear regression, logistic regression is also used to estimate the relationship between a dependent variable and one or more independent variables, but it is used to make a prediction about a categorical variable versus a continuous one. A categorical variable can be true or false, yes or no, 1 or 0, et cetera. The unit of measure also differs from linear regression as it produces a probability, but the logit function transforms the S-curve into straight line.  

Random Forest Classifier: An ensemble learning method that uses multiple decision trees for classification, resulting in robust performance in the presence of noise and non-linearity in data. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. Trees in the forest use the best split strategy, i.e. equivalent to passing splitter="best" to the underlying DecisionTreeRegressor. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

## Model Evaluation:
Both models are evaluated based on key metrics such as accuracy, precision, recall, and F1-score. The model with the highest F1-score is selected as the final model.

Accuracy is the proportion of all classifications that were correct, whether positive or negative.

The true positive rate (TPR), or the proportion of all actual positives that were classified correctly as positives, is also known as recall.

The false positive rate (FPR) is the proportion of all actual negatives that were classified incorrectly as positives, also known as the probability of false alarm.

Precision is the proportion of all the model's positive classifications that are actually positive.

The F1 score is the harmonic mean (a kind of average) of precision and recall.
## Project Structure
### SentimentAnalysis Class:

#### Attributes:

data_path: Path to the CSV file containing the reviews dataset.

data: Loaded DataFrame with reviews and sentiments.

tfidf_matrix: TF-IDF matrix representing the reviews.

vectorizer: TF-IDF vectorizer instance.

X_train, X_test, y_train, y_test: Training and testing datasets.

log_reg: Logistic Regression model instance.

random_forest: Random Forest model instance.


#### Methods:

load_data(): Loads data from the provided path and mounts the Google Drive using Colab.

preprocess_text(text): Cleans and preprocesses individual review texts by removing noise and applying lemmatization.

preprocess_data(): Applies text preprocessing to the entire dataset.

vectorize_data(): Converts preprocessed text into a numerical form using the TF-IDF vectorizer.

split_data(): Splits the dataset into training and test sets.

train_models(): Trains both Logistic Regression and Random Forest models.

evaluate_model(true, pred): Computes the accuracy, precision, recall, and F1-score for a given set of true and predicted labels.

evaluate_models(): Compares the performance of Logistic Regression and Random Forest models and selects the one with the higher F1-score.


### Execution Flow:

The dataset is loaded and preprocessed.

Text data is transformed into feature vectors using TF-IDF.

The dataset is split into training and test sets.

Both models are trained and evaluated on the test set.

Performance metrics are printed for both models, and the better model is selected based on the F1-score.

## Dataset
The input dataset should be a CSV file containing at least two columns:

review: A text column containing product reviews.

sentiment: A target column labeled as either "positive" or "negative", which will be converted to binary values (1 for positive, 0 for negative).

## Results
After running the program, you will see the performance of both models in terms of accuracy, precision, recall, and F1-score. The final model is selected based on the highest F1-score, ensuring a balance between precision and recall.
## References:
NLTK

Python

IBM

GeeksforGeeks
