"""MLP model for continuous valence/arousal prediction."""
import torch
import torch.nn as nn

class ValenceArousalMLP(nn.Module):
    """MLP for predicting continuous valence and arousal."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = None,
                 use_tanh: bool = True,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input latents (d_vae_latent)
            hidden_dim: Hidden layer dimension (default: input_dim // 2)
            use_tanh: Whether to use tanh activation to constrain outputs to [-1, 1]
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.use_tanh = use_tanh
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 2 outputs: valence, arousal
        )
        
        if use_tanh:
            self.tanh = nn.Tanh()
    
    def forward(self, latents, mask=None):
        """
        Args:
            latents: (batch_size, seq_len, input_dim) or (batch_size, input_dim) if pooled
            mask: (batch_size, seq_len) mask for valid positions
        
        Returns:
            predictions: (batch_size, seq_len, 2) or (batch_size, 2) if pooled
        """
        if len(latents.shape) == 3:
            # Sequence-level: average over valid positions
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                latents = (latents * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            else:
                latents = latents.mean(dim=1)
        
        output = self.mlp(latents)
        
        if self.use_tanh:
            output = self.tanh(output)
        
        return output