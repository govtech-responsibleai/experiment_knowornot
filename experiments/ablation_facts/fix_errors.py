import pandas as pd
import numpy as np
import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# # Download stopwords if not already present
# nltk.download('stopwords')

# Preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)


def filter_keyword_duplicates(
    df: pd.DataFrame, diversity_threshold: float = 0.3
) -> pd.DataFrame:
    if df.empty:
        return df

    # Preprocess questions
    processed_questions = df["question"].astype(str).apply(preprocess_text)

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(strip_accents="unicode", min_df=1, max_df=0.9, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(processed_questions)
    uniqueness_scores = np.sum(tfidf_matrix.toarray(), axis=1)

    # Sort questions by uniqueness score (most unique first)
    sorted_indices = np.argsort(-uniqueness_scores)

    # Calculate threshold
    min_score = uniqueness_scores.min()
    max_score = uniqueness_scores.max()
    score_range = max_score - min_score
    threshold_value = min_score + (score_range * diversity_threshold)

    # Select rows above threshold
    selected_indices = [
        idx for idx in sorted_indices if uniqueness_scores[idx] >= threshold_value
    ]
    filtered_df = df.iloc[selected_indices].reset_index(drop=True)

    print(f"Selected {len(filtered_df)} out of {len(df)} questions based on TF-IDF uniqueness.")

    return filtered_df

def filter_semantic_duplicates(
    df: pd.DataFrame, min_distance: float = 0.3
) -> pd.DataFrame:
    if df.empty:
        return df

    texts = df["question"].astype(str).tolist()
    embeddings = get_embeddings(texts)
    embeddings_array = np.array(embeddings)

    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized = embeddings_array / norms

    selected_indices = [0]
    remaining_indices = list(range(1, len(df)))

    while remaining_indices:
        max_min_distance = -1
        best_idx = -1

        for idx in remaining_indices:
            distances = [
                1.0 - np.dot(normalized[idx], normalized[sel_idx])
                for sel_idx in selected_indices
            ]
            min_dist = min(distances)
            if min_dist > max_min_distance:
                max_min_distance = min_dist
                best_idx = idx

        if max_min_distance >= min_distance:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break

    print(f"Semantic filtering: Selected {len(selected_indices)} out of {len(df)} questions")
    return df.iloc[selected_indices].reset_index(drop=True)



def main():
    input_file = "experiments/ablation_facts/questions_csv/BTT/labelled/BTT_filtered_qa_sem0.10.csv"
    output_file = "experiments/ablation_facts/questions_csv/BTT/labelled/BTT_filtered_qa_sem0.10_kwfilter.csv"

    df = pd.read_csv(input_file)
    filtered_df = filter_keyword_duplicates(df, diversity_threshold=0.3)
    filtered_df.to_csv(output_file, index=False)

    print(f"Filtered results saved to '{output_file}'")


if __name__ == "__main__":
    main()
