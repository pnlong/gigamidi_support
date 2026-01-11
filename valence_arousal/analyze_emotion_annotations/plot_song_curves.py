"""
Create valence/arousal curves (bar-by-bar) for example songs.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random

def plot_song_curves(annotations_path: str, output_path: str, n_examples: int = 5, seed: int = None):
    """Create valence/arousal curves for random songs."""
    if seed is not None:
        random.seed(seed)
    
    df = pd.read_csv(annotations_path)
    
    # Get unique songs
    unique_songs = df['md5'].unique()
    n_examples = min(n_examples, len(unique_songs))
    example_songs = random.sample(list(unique_songs), n_examples)
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for idx, md5 in enumerate(example_songs):
        song_df = df[df['md5'] == md5].sort_values('bar_number')
        
        ax = axes[idx]
        ax.plot(song_df['bar_number'], song_df['valence'], 
                marker='o', label='Valence', linewidth=2)
        ax.plot(song_df['bar_number'], song_df['arousal'], 
                marker='s', label='Arousal', linewidth=2)
        ax.set_xlabel('Bar Number')
        ax.set_ylabel('Value')
        ax.set_title(f'Song {md5[:8]}... (Valence/Arousal over time)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved song curves to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    parser.add_argument("--output_path", required=True,
                       help="Output file path for song curves plot")
    parser.add_argument("--n_examples", type=int, default=5,
                       help="Number of example songs to plot")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for song selection")
    args = parser.parse_args()
    
    plot_song_curves(args.annotations_path, args.output_path, args.n_examples, args.seed)