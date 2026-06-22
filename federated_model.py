import torch
import torch.nn as nn
import torch.nn.functional as F


class ParkinsonNN(nn.Module):
    """
    Upgraded Federated Neural Network for Parkinson's Detection.

    Improvements over baseline (3 plain Linear layers):

    1. BatchNorm1d after each Linear layer
       — Normalizes activations within each mini-batch
       — Reduces internal covariate shift
       — Especially important in Federated Learning where each
         client sees different data distributions (Non-IID)

    2. Dropout regularization
       — Randomly zeroes neurons during training
       — Prevents overfitting on small client datasets
         (Client 2 has only ~228 samples — very prone to overfit)

    3. Wider first layer (128 → 256)
       — Gives the model more capacity to learn complex
         feature interactions from 128 input features

    4. Residual (skip) connection
       — Connects block1 output directly to block2 output
       — Helps gradient flow during backpropagation
       — Prevents vanishing gradients in deeper layers
       — Borrowed from ResNet — standard in medical ML

    5. Kaiming weight initialization
       — Designed specifically for ReLU activations
       — Ensures stable training from round 1 in FL
       — Random init causes slower convergence across rounds

    Architecture:
        Input(128)
            → Block1: Linear(128→256) → BN → ReLU → Dropout(0.3)
            → Block2: Linear(256→128) → BN → ReLU → Dropout(0.3)
            + Skip:   Linear(128→128) [projects input to match Block2 output]
            → Block3: Linear(128→64)  → BN → ReLU → Dropout(0.2)
            → Output: Linear(64→2)
    """

    def __init__(self, input_size=149, dropout1=0.3, dropout2=0.2):
        super(ParkinsonNN, self).__init__()

        # ---- Block 1 ----
        self.block1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout1)
        )

        # ---- Block 2 ----
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout1)
        )

        # ---- Skip connection projection ----
        # Projects input (128) → 128 to match block2 output shape
        # Allows residual addition: out = block2(block1(x)) + skip(x)
        self.skip = nn.Linear(input_size, 128)

        # ---- Block 3 ----
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout2)
        )

        # ---- Output layer ----
        self.output = nn.Linear(64, 2)

        # ---- Weight initialization ----
        self._initialize_weights()


    def _initialize_weights(self):
        """
        Kaiming (He) initialization for all Linear layers.
        Designed for ReLU — keeps variance stable across layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):

        # Block 1
        out = self.block1(x)

        # Block 2 + residual skip
        out = self.block2(out) + self.skip(x)

        # Block 3
        out = self.block3(out)

        # Output (raw logits — CrossEntropyLoss handles softmax)
        out = self.output(out)

        return out


    def get_architecture_summary(self):
        """
        Prints a clean summary of the architecture.
        Call this before training to verify the model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "="*50)
        print("  ParkinsonNN Architecture Summary")
        print("="*50)
        print(f"  Input size       : 149 features (128 acoustic + 21 clinical)")
        print(f"  Block 1          : Linear(128→256) + BN + ReLU + Dropout(0.3)")
        print(f"  Block 2          : Linear(256→128) + BN + ReLU + Dropout(0.3)")
        print(f"  Skip connection  : Linear(128→128)")
        print(f"  Block 3          : Linear(128→64)  + BN + ReLU + Dropout(0.2)")
        print(f"  Output           : Linear(64→2)")
        print(f"  Total params     : {total_params:,}")
        print(f"  Trainable params : {trainable:,}")
        print("="*50 + "\n")


# ============================================================
# Quick sanity check — run this file directly to verify
# ============================================================
if __name__ == "__main__":

    model = ParkinsonNN()
    model.get_architecture_summary()

    # Test forward pass with dummy batch
    dummy_input = torch.randn(16, 149)   # batch of 16, 149 features each
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape  : {dummy_input.shape}")
    print(f"  Output shape : {output.shape}")
    print(f"  Output (first 3 rows):\n{output[:3]}")
    print("\n  Sanity check passed.")