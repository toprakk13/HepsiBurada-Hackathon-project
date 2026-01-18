Hybrid Address Classification

This project is a hybrid Natural Language Processing (NLP) pipeline developed to match irregular and noisy address texts (misspellings, abbreviations, etc.) with standardized neighborhood names. It combines deep learning–based Named Entity Recognition (NER) and Vector Space Similarity methods.

About the Project

Address data is often entered by users as free text and does not follow a standard format. This project takes such unstructured address texts and matches them with the correct records from an official neighborhood list.

Approach Used:
To improve accuracy, the project follows a two-stage waterfall approach:
	1.	NER (Named Entity Recognition): Attempts to directly extract the neighborhood name from the address text.
	2.	Vector Search (k-NN): If NER fails to produce a result, the overall structure of the address is compared with similar addresses in the training data.

 Technical Architecture

The project consists of four main stages:

1. Data Preprocessing
	•	Addresses are converted to lowercase and normalized to ASCII characters.
	•	Common abbreviations (e.g., “Mah.”, “Sk.”) are expanded to their full forms (“Mahallesi”, “Sokak”).

2. Entity Extraction (via NER)
	•	Model: The savasy/bert-base-turkish-ner-cased model from Hugging Face is used.
	•	Words labeled as LOC (Location) are extracted from the address text.
	•	The extracted candidate is matched to the closest official neighborhood name using RapidFuzz for fuzzy matching.

3. Similarity Search (FAISS & TF-IDF)
	•	Training addresses are vectorized using a TF-IDF Vectorizer (n-grams: 1–3).
	•	FAISS (Facebook AI Similarity Search) is used to perform fast k-Nearest Neighbors (k-NN) search in high-dimensional space.
	•	The top 3 most similar training addresses to the test address are retrieved, and the final label is determined via majority voting.

4. Hybrid Decision Mechanism
	•	NER output has priority. If a neighborhood name is successfully extracted, this result is used.
	•	If no entity is found by NER, the k-NN (FAISS) result is used instead.
