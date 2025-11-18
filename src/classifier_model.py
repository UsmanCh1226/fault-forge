import torch.nn as nn
# Import the constant to ensure the input size is consistent (420 in your case)
try:
    from vae_time_series import INPUT_DIM
except ImportError:
    # Fallback/Error check
    INPUT_DIM = 420 

class HealthClassifier(nn.Module):
    """
    A simple MLP classifier for predicting Normal (0) or Anomaly (1) health state.
    It takes the flattened time-series sequence as input.
    """
    def __init__(self, input_dim=INPUT_DIM):
        super(HealthClassifier, self).__init__()
        
        # The first layer must match the INPUT_DIM (420)
        self.network = nn.Sequential(
            # Input Layer (INPUT_DIM -> 256)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden Layer (256 -> 128)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output Layer (128 -> 1). Sigmoid will be applied in the loss function.
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        # Flatten the input just in case it wasn't pre-flattened (though it should be)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

if __name__ == '__main__':
    # Example usage test
    import torch
    model = HealthClassifier()
    print(f"Classifier created with Input Dimension: {INPUT_DIM}")
    # Simulate a batch of data
    dummy_input = torch.randn(10, INPUT_DIM) 
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")