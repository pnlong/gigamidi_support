"""
Create bar charts for emotion and genre distributions in GigaMIDI annotations.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_histograms(annotations_path: str, output_path: str):
    """Create bar charts for emotion and genre distributions."""
    df = pd.read_csv(annotations_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Emotion distribution
    emotion_counts = df['emotion'].value_counts().sort_index()
    axes[0].bar(range(len(emotion_counts)), emotion_counts.values, edgecolor='black', alpha=0.7)
    axes[0].set_xticks(range(len(emotion_counts)))
    axes[0].set_xticklabels(emotion_counts.index, rotation=45, ha='right')
    axes[0].set_xlabel('Emotion')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Emotion Distribution')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Genre distribution
    genre_counts = df['genre'].value_counts().sort_index()
    axes[1].bar(range(len(genre_counts)), genre_counts.values, edgecolor='black', alpha=0.7)
    axes[1].set_xticks(range(len(genre_counts)))
    axes[1].set_xticklabels(genre_counts.index, rotation=45, ha='right')
    axes[1].set_xlabel('Genre')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Genre Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histograms to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    parser.add_argument("--output_path", required=True,
                       help="Output file path for histogram plot")
    args = parser.parse_args()
    
    plot_histograms(args.annotations_path, args.output_path)
