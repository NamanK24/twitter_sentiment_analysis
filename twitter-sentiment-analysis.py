import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class NaiveBayesClassifier:
    def __init__(self):
        self.class_word_probs = {}
        self.class_priors = {}
        self.classes = []

    def train(self, X_train, y_train):
        self.classes = np.unique(y_train)
        total_samples = len(y_train)
        for class_ in self.classes:
            class_samples = X_train[y_train == class_]
            self.class_priors[class_] = len(class_samples) / total_samples
            all_words = [word for tweet in class_samples for word in tweet.split()]
            word_counts = pd.Series(all_words).value_counts()
            self.class_word_probs[class_] = (word_counts + 1) / (
                len(all_words) + len(word_counts)
            )

    def predict(self, X_test):
        predictions = []
        for tweet in X_test:
            tweet_words = tweet.split()
            probs = {
                class_: np.log(self.class_priors[class_])
                + sum(
                    np.log(self.class_word_probs[class_].get(word, 1e-10))
                    for word in tweet_words
                )
                for class_ in self.classes
            }
            predicted_class = max(probs, key=probs.get)
            predictions.append(predicted_class)
        return predictions


def main():
    st.title("Twitter Sentiment Analysis")

    data = pd.read_csv("tweets-train.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        data["tweet"], data["label"], test_size=0.15, random_state=42
    )

    clf = NaiveBayesClassifier()
    clf.train(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    st.write("## Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2f}")

    st.write("### Classification Report")
    st.write(classification_report(y_test, y_pred))

    keyword = st.text_input("Enter the keyword to search:")

    if keyword:
        search_results = data[data["tweet"].str.contains(keyword, case=False)]
        st.write("### Search Results")
        st.write(search_results)

    st.write("## Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(plt)


if __name__ == "__main__":
    main()
