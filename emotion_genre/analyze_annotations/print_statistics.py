"""
Print summary statistics about GigaMIDI annotations.
"""

import argparse
import pandas as pd
from collections import Counter

def print_statistics(annotations_path: str):
    """Print summary statistics."""
    df = pd.read_csv(annotations_path)
    
    print("\n" + "="*60)
    print("GigaMIDI Annotation Statistics")
    print("="*60)
    
    print(f"\nTotal songs annotated: {len(df)}")
    print(f"Unique songs: {df['md5'].nunique()}")
    
    print("\n" + "-"*60)
    print("Emotion Distribution:")
    print("-"*60)
    emotion_counts = df['emotion'].value_counts().sort_index()
    emotion_pct = df['emotion'].value_counts(normalize=True).sort_index() * 100
    for emotion in emotion_counts.index:
        print(f"  {emotion:15s}: {emotion_counts[emotion]:6d} ({emotion_pct[emotion]:5.2f}%)")
    
    print("\n" + "-"*60)
    print("Genre Distribution:")
    print("-"*60)
    genre_counts = df['genre'].value_counts().sort_index()
    genre_pct = df['genre'].value_counts(normalize=True).sort_index() * 100
    for genre in genre_counts.index:
        print(f"  {genre:15s}: {genre_counts[genre]:6d} ({genre_pct[genre]:5.2f}%)")
    
    print("\n" + "-"*60)
    print("Prediction Confidence:")
    print("-"*60)
    print(f"  Emotion - Mean: {df['emotion_prob'].mean():.4f}, Std: {df['emotion_prob'].std():.4f}")
    print(f"  Emotion - Min: {df['emotion_prob'].min():.4f}, Max: {df['emotion_prob'].max():.4f}")
    print(f"  Genre - Mean: {df['genre_prob'].mean():.4f}, Std: {df['genre_prob'].std():.4f}")
    print(f"  Genre - Min: {df['genre_prob'].min():.4f}, Max: {df['genre_prob'].max():.4f}")
    
    print("\n" + "-"*60)
    print("Most Common Emotion-Genre Combinations:")
    print("-"*60)
    combinations = df.groupby(['emotion', 'genre']).size().sort_values(ascending=False).head(10)
    for (emotion, genre), count in combinations.items():
        pct = (count / len(df)) * 100
        print(f"  {emotion:15s} + {genre:10s}: {count:6d} ({pct:5.2f}%)")
    
    print("\n" + "-"*60)
    print("High Confidence Predictions:")
    print("-"*60)
    high_conf_threshold = 0.8
    high_conf_emotion = (df['emotion_prob'] >= high_conf_threshold).sum()
    high_conf_genre = (df['genre_prob'] >= high_conf_threshold).sum()
    print(f"  Emotion predictions with prob >= {high_conf_threshold}: {high_conf_emotion} ({high_conf_emotion/len(df)*100:.2f}%)")
    print(f"  Genre predictions with prob >= {high_conf_threshold}: {high_conf_genre} ({high_conf_genre/len(df)*100:.2f}%)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    args = parser.parse_args()
    
    print_statistics(args.annotations_path)
