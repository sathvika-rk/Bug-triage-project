import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def run_naive_bayes_baseline():
    # 1. Load the data generated in Phase 1
    try:
        df = pd.read_csv("cleaned_issues.csv")
    except FileNotFoundError:
        print("Error: cleaned_issues.csv not found. Please run your scraper script first!")
        return

    # Clean out rows that have absolutely no labels (our targets)
    df = df.dropna(subset=['labels'])

    # Safely fill missing text with empty strings so pandas doesn't create NaNs
    df['title'] = df['title'].fillna("")
    df['body'] = df['body'].fillna("")

    # 2. Extract Features and Labels
    # Combine Title and Body to give the model full context
    X = df['title'] + " " + df['body']
    y = df['labels']

    # Split into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split successfully.")
    print(f"Vectorizing {len(X_train)} training reports and {len(X_test)} testing reports...\n")

    # 3. Text Vectorization via TF-IDF
    # Convert words to numerical features, ignoring common English stop words
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Instantiate and Train the Multinomial Naive Bayes Classifier
    print("Training Multinomial Naive Bayes model...")
    nb_model = MultinomialNB(alpha=1.0) # alpha=1.0 enables Laplace smoothing
    nb_model.fit(X_train_tfidf, y_train)

    # 5. Evaluate the Model
    predictions = nb_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    # 6. Output Detailed Metrics
    print("\n==================================================")
    print("      MULTINOMIAL NAIVE BAYES BASELINE RESULTS     ")
    print("==================================================")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    
    # zero_division=0 prevents the script from throwing warnings if a niche label is never predicted
    print(classification_report(y_test, predictions, zero_division=0))
    print("==================================================")
    
    # Save a reference file of the metrics for your resume records
    with open("baseline_results.txt", "w") as f:
        f.write(f"Baseline Algorithm: Multinomial Naive Bayes\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(classification_report(y_test, predictions, zero_division=0))
    print("Results saved to baseline_results.txt for future resume verification.")

if __name__ == "__main__":
    run_naive_bayes_baseline()