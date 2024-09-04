import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

nltk.download('punkt')

# Define the book labels
y_book_array = ["HP1", "HP2", "HP3", "HP4", "HP5", "HP6", "HP7"]

# Function to read and preprocess text with error handling
def read_and_preprocess(files):
    corpus = []
    book_tokens = []
   
    for file in files:
        if os.path.exists(file):
            with open(file, 'r') as f:
                text = f.read().lower()
                # Split text into pages
                pages = text.split('\n')
                
                book_pages_tokens = []
                for page in pages:
                    # Remove punctuation
                    page = re.sub(r'[^\w\s]', '', page)
                    # Tokenize
                    tokens = word_tokenize(page)
                    corpus.extend(tokens)
                    book_pages_tokens.append(tokens)
                
                book_tokens.append(book_pages_tokens)
        else:
            print(f"Error: File not found - {file}")
    return corpus, book_tokens

# Specify the correct absolute or relative paths to your files
# files = [
#     '/Users/manisha/Downloads/NLP/Lab2/HarryPotter/HP1.txt',
#     '/Users/manisha/Downloads/NLP/Lab2/HarryPotter/HP2.txt',
#     '/Users/manisha/Downloads/NLP/Lab2/HarryPotter/HP3.txt',
#     '/Users/manisha/Downloads/NLP/Lab2/HarryPotter/HP4.txt',
#     '/Users/manisha/Downloads/NLP/Lab2/HarryPotter/HP5.txt',
#     '/Users/manisha/Downloads/NLP/Lab2/HarryPotter/HP6.txt',
#     '/Users/manisha/Downloads/NLP/Lab2/HarryPotter/HP7.txt'
# ]
files = ['HarryPotter/HP1.txt', 'HarryPotter/HP2.txt', 'HarryPotter/HP3.txt', 'HarryPotter/HP4.txt','HarryPotter/HP5.txt', 'HarryPotter/HP6.txt', 'HarryPotter/HP7.txt']

# Preprocess the text
corpus, tokens = read_and_preprocess(files)

def extract_ngrams_from_tokens(tokens, book, n):
    n_grams = []
    y_array = []
    for i in range(len(tokens) - n + 1):
        n_gram = ' '.join(tokens[i:i+n])
        n_grams.append(n_gram)
        y_array.append(book)
    return n_grams, y_array

# Function to generate trigrams and labels for given n
def generate_ngrams(tokens, n):
    x_y_trigrams = pd.DataFrame(columns=['trigrams', 'book'])
    for i in range(len(files)):
        pages_trigramed = []
        pages_y_array = []
        for j in range(len(tokens[i])): 
            trigrams, trigrams_y_array = extract_ngrams_from_tokens(tokens[i][j], y_book_array[i], n)
            pages_trigramed.append(trigrams)
            pages_y_array.append(y_book_array[i])
        book_x_y_trigrams = pd.DataFrame({"trigrams": pages_trigramed, "book": pages_y_array})
        x_y_trigrams = pd.concat([x_y_trigrams, book_x_y_trigrams], ignore_index=True)
    return x_y_trigrams

# Define n-gram and delta values to test
n_values = [1, 3]
deltas = [0.1, 1.0]

# Iterate over combinations of n-values and deltas
for n in n_values:
    print(f"\nEvaluating with n-grams = {n}")
    x_y_trigrams = generate_ngrams(tokens, n)
    
    # Split the data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        x_y_trigrams["trigrams"], x_y_trigrams["book"], test_size=0.1, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42)

    def calculate_class_weights(y_train):
        classes = np.unique(y_train)
        class_weights = compute_class_weight(None, classes=classes, y=y_train)
        return dict(zip(classes, class_weights))

    class_weights = calculate_class_weights(y_train)

    def get_priors(y_train):
        book_counts = Counter(y_train)
        total_count = len(y_train)
        priors = {i: book_counts[i]/total_count for i in book_counts}
        return priors

    def get_likelihood(book: str, n_gram: str, n_gram_map: Dict[str, Dict[str, int]], 
                       total_trigrams_per_class: Dict[str, int], vocab_size: int, delta: float) -> float:
        numerator = delta + n_gram_map.get(n_gram, {}).get(book, 0)
        denominator = delta * vocab_size + total_trigrams_per_class[book]
        return numerator / denominator

    def n_gram_mapping(x_train: List[List[str]], y_train: List[str], classes: List[str]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        n_gram_map = defaultdict(lambda: defaultdict(int))
        total_trigrams_per_class = {c: 0 for c in classes}
        for trigrams, book in zip(x_train, y_train):
            for n_gram in trigrams:
                n_gram_map[n_gram][book] += 1
                total_trigrams_per_class[book] += 1
        return n_gram_map, total_trigrams_per_class

    def prob_page_in_book(page, priors, n_gram_map, total_trigrams_per_class, classes, vocab_size, class_weights, delta):
        log_probs = []
        for c in classes:
            log_prob = np.log(priors[c] * class_weights[c])
            for n_gram in page:
                likelihood = get_likelihood(c, n_gram, n_gram_map, total_trigrams_per_class, vocab_size, delta)
                log_prob += np.log(likelihood)
            log_probs.append(log_prob)
        return log_probs

    def predict(X: List[List[str]], y: List[str], X_test: List[List[str]], classes: List[str], delta: float) -> List[str]:
        priors = get_priors(y)
        n_gram_map, total_trigrams_per_class = n_gram_mapping(X, y, classes)
        vocab_size = len(set(n_gram for page in X for n_gram in page))
        class_weights = calculate_class_weights(y)
        predictions = []
        for page in X_test:
            log_probs = prob_page_in_book(page, priors, n_gram_map, total_trigrams_per_class, classes, vocab_size, class_weights, delta)
            predictions.append(classes[np.argmax(log_probs)])
        return predictions

    # Calculate accuracies for each delta value
    for delta in deltas:
        print(f"\nEvaluating with delta = {delta}")
        
        # Predictions for validation set
        predictions_val = predict(X_train, y_train, X_val, y_book_array, delta)
        accuracy_val = sum(p == a for p, a in zip(predictions_val, y_val)) / len(y_val)
        print(f"Accuracy - Validation (n = {n}, Delta = {delta}): {accuracy_val * 100:.2f}%")

        # Predictions for test set
        predictions_test = predict(X_train, y_train, X_test, y_book_array, delta)
        accuracy_test = sum(p == a for p, a in zip(predictions_test, y_test)) / len(y_test)
        print(f"Accuracy - Test (n = {n}, Delta = {delta}): {accuracy_test * 100:.2f}%")
