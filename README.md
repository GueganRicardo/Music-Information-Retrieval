# Music-Information-Retrieval

## Introduction
This project explores the functioning of musical recommendation systems, focusing on analyzing audio signals and metadata to identify emotions in songs.

## Exercise 1: Spotify Recommendation System Analysis
We learned that Spotify's system relies on:
- **User Data**: History and favorites.
- **Collaborative Filtering**: Recommendations based on similar profiles.
- **Audio Analysis**: Searching for songs with similar features.

## Exercise 2: Audio Feature Extraction
We used the Librosa library to extract information such as:
- **Timbre**: Mel-Frequency Cepstral Coefficients and others.
- **Intensity**: Root Mean Square.
- **Frequency**: Fundamental frequency and zero-crossing rate.
- **Rhythm**: Tempo.

## Exercise 3: Similarity Rankings
We compared songs using Euclidean, Manhattan, and cosine distances, normalizing data to ensure equal weight in comparisons.

## Exercise 4: Metadata-Based Recommendations
We implemented a scoring system for data like Artist, Emotion/Mood, and Genre, comparing results with audio features.

## Conclusion
The results show that combining metadata-based and content-based recommendations can enhance the accuracy of musical suggestions.
