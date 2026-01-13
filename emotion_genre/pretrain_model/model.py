"""MLP classifier for emotion/genre recognition."""
import torch
import torch.nn as nn

class EmotionGenreClassifier(nn.Module):
    """MLP classifier for emotion or genre recognition."""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dim: int = None,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input latents (d_vae_latent)
            num_classes: Number of classes (11 for emotion, 6 for genre)
            hidden_dim: Hidden layer dimension (default: input_dim // 2)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # Classification head
        )
    
    def forward(self, latents):
        """
        Args:
            latents: (batch_size, input_dim) - already pooled
        
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.mlp(latents)
