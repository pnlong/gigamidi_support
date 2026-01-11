"""
Create boxplots for valence and arousal.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_boxplots(annotations_path: str, output_path: str):
    """Create boxplots for valence and arousal."""
    df = pd.read_csv(annotations_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].boxplot(df['valence'], vert=True)
    axes[0].set_ylabel('Valence')
    axes[0].set_title('Valence Boxplot')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(df['arousal'], vert=True)
    axes[1].set_ylabel('Arousal')
    axes[1].set_title('Arousal Boxplot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved boxplots to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    parser.add_argument("--output_path", required=True,
                       help="Output file path for boxplot")
    args = parser.parse_args()
    
    plot_boxplots(args.annotations_path, args.output_path)