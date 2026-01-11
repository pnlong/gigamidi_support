"""
Print summary statistics about GigaMIDI annotations.
"""

import argparse
import pandas as pd

def print_statistics(annotations_path: str):
    """Print summary statistics."""
    df = pd.read_csv(annotations_path)
    
    print("\nSummary Statistics:")
    print(f"Total bars: {len(df)}")
    print(f"Total songs: {df['md5'].nunique()}")
    print(f"Average bars per song: {len(df) / df['md5'].nunique():.2f}")
    print(f"\nValence:")
    print(df['valence'].describe())
    print(f"\nArousal:")
    print(df['arousal'].describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    args = parser.parse_args()
    
    print_statistics(args.annotations_path)