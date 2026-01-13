"""
Create visualizations comparing emotion/genre predictions by GigaMIDI's original genre metadata.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import load_dataset

def load_gigamidi_metadata(split='train', streaming=True):
    """
    Load GigaMIDI metadata to get genre information.
    Returns dict mapping md5 to metadata (including music_styles_curated).
    """
    metadata = {}
    try:
        if streaming:
            dataset = load_dataset("Metacreation/GigaMIDI", split=split, streaming=True)
        else:
            dataset = load_dataset("Metacreation/GigaMIDI", split=split)
        
        for sample in dataset:
            md5 = sample.get("md5", "")
            if md5:
                metadata[md5] = {
                    "music_styles_curated": sample.get("music_styles_curated", []),
                    "title": sample.get("title", ""),
                    "artist": sample.get("artist", ""),
                }
    except Exception as e:
        print(f"Warning: Could not load GigaMIDI metadata: {e}")
    return metadata

def plot_by_genre(annotations_path: str, output_path: str, top_n: int = 10, split: str = 'train', streaming: bool = True):
    """Create visualizations comparing predictions by GigaMIDI genre."""
    df = pd.read_csv(annotations_path)
    
    # Load metadata
    print("Loading GigaMIDI metadata...")
    metadata = load_gigamidi_metadata(split=split, streaming=streaming)
    
    if not metadata:
        print("Error: Could not load metadata. Cannot create genre comparison plots.")
        return
    
    print(f"Loaded metadata for {len(metadata)} songs")
    
    # Merge metadata with annotations
    df_with_genre = df.copy()
    df_with_genre['gigamidi_genres'] = df_with_genre['md5'].map(
        lambda x: metadata.get(x, {}).get('music_styles_curated', [])
    )
    
    # Explode genres (songs can have multiple genres)
    df_exploded = df_with_genre.explode('gigamidi_genres')
    df_exploded = df_exploded[df_exploded['gigamidi_genres'].notna()]
    
    if len(df_exploded) == 0:
        print("Warning: No genre information available. Cannot create genre comparison plots.")
        return
    
    # Get top N genres by count
    top_genres = df_exploded['gigamidi_genres'].value_counts().head(top_n).index.tolist()
    df_top = df_exploded[df_exploded['gigamidi_genres'].isin(top_genres)]
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Emotion distribution by GigaMIDI genre (heatmap)
    emotion_by_genre = pd.crosstab(df_top['gigamidi_genres'], df_top['emotion'])
    # Normalize by row to show proportions
    emotion_by_genre_norm = emotion_by_genre.div(emotion_by_genre.sum(axis=1), axis=0)
    
    sns.heatmap(emotion_by_genre_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0],
               xticklabels=True, yticklabels=True)
    axes[0].set_title('Emotion Distribution by GigaMIDI Genre (Normalized)')
    axes[0].set_xlabel('Predicted Emotion')
    axes[0].set_ylabel('GigaMIDI Genre')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Genre prediction vs GigaMIDI genre (confusion matrix style)
    genre_by_genre = pd.crosstab(df_top['gigamidi_genres'], df_top['genre'])
    # Normalize by row
    genre_by_genre_norm = genre_by_genre.div(genre_by_genre.sum(axis=1), axis=0)
    
    sns.heatmap(genre_by_genre_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[1],
               xticklabels=True, yticklabels=True)
    axes[1].set_title('Predicted Genre Distribution by GigaMIDI Genre (Normalized)')
    axes[1].set_xlabel('Predicted Genre')
    axes[1].set_ylabel('GigaMIDI Genre')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved genre comparison plots to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    parser.add_argument("--output_path", required=True,
                       help="Output file path for genre comparison plot")
    parser.add_argument("--top_n", type=int, default=10,
                       help="Number of top genres to include")
    parser.add_argument("--split", type=str, default="train",
                       help="GigaMIDI split to load metadata from")
    parser.add_argument("--no_streaming", action="store_true",
                       help="Disable streaming mode for metadata loading")
    args = parser.parse_args()
    
    plot_by_genre(args.annotations_path, args.output_path, args.top_n, args.split, not args.no_streaming)
