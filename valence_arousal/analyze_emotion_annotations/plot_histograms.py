"""
Create histograms for valence and arousal distributions.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_histograms(annotations_path: str, output_path: str, bins: int = 50):
    """Create histograms for valence and arousal."""
    df = pd.read_csv(annotations_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(df['valence'], bins=bins, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Valence')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Valence Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(df['arousal'], bins=bins, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Arousal')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Arousal Distribution')
    axes[1].grid(True, alpha=0.3)
    
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
    parser.add_argument("--bins", type=int, default=50,
                       help="Number of bins for histograms")
    args = parser.parse_args()
    
    plot_histograms(args.annotations_path, args.output_path, args.bins)