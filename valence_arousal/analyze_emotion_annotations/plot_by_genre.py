"""
Create boxplots comparing valence/arousal by genre.
Requires GigaMIDI metadata to be loaded.
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
    """Create boxplots comparing valence/arousal by genre."""
    df = pd.read_csv(annotations_path)
    
    # Load metadata
    print("Loading GigaMIDI metadata...")
    metadata = load_gigamidi_metadata(split=split, streaming=streaming)
    
    if not metadata:
        print("Error: Could not load metadata. Cannot create genre plots.")
        return
    
    print(f"Loaded metadata for {len(metadata)} songs")
    
    # Merge metadata with annotations
    df_with_genre = df.copy()
    df_with_genre['genres'] = df_with_genre['md5'].map(
        lambda x: metadata.get(x, {}).get('music_styles_curated', [])
    )
    
    # Explode genres (songs can have multiple genres)
    df_exploded = df_with_genre.explode('genres')
    df_exploded = df_exploded[df_exploded['genres'].notna()]
    
    if len(df_exploded) == 0:
        print("Warning: No genre information available. Cannot create genre plots.")
        return
    
    # Get top N genres by count
    top_genres = df_exploded['genres'].value_counts().head(top_n).index.tolist()
    df_top = df_exploded[df_exploded['genres'].isin(top_genres)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Valence by genre
    sns.boxplot(data=df_top, x='genres', y='valence', ax=axes[0])
    axes[0].set_title('Valence by Genre')
    axes[0].set_xlabel('Genre')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Arousal by genre
    sns.boxplot(data=df_top, x='genres', y='arousal', ax=axes[1])
    axes[1].set_title('Arousal by Genre')
    axes[1].set_xlabel('Genre')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved genre boxplots to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    parser.add_argument("--output_path", required=True,
                       help="Output file path for genre boxplot")
    parser.add_argument("--top_n", type=int, default=10,
                       help="Number of top genres to include")
    parser.add_argument("--split", type=str, default="train",
                       help="GigaMIDI split to load metadata from")
    parser.add_argument("--no_streaming", action="store_true",
                       help="Disable streaming mode for metadata loading")
    args = parser.parse_args()
    
    plot_by_genre(args.annotations_path, args.output_path, args.top_n, args.split, not args.no_streaming)