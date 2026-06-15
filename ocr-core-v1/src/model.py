"""
CRNN model for S.C.O.R.E. handwriting recognition.
Architecture: CNN backbone → BiLSTM sequence modeller → linear classifier
Output shape: (T, N, num_classes) — required by torch.nn.CTCLoss
"""

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """Single BiLSTM layer with a linear projection."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, input_size)
        out, _ = self.rnn(x)          # (T, N, hidden*2)
        return self.fc(out)           # (T, N, output_size)


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for handwriting recognition.

    Input:  (N, 1, H, W)  — grayscale image, H=32 recommended
    Output: (T, N, num_classes) — log-softmax NOT applied here;
            apply F.log_softmax before passing to CTCLoss.

    Args:
        num_classes: size of the character vocabulary + 1 for CTC blank token
        img_height:  height images are resized to before the CNN (default 32)
        lstm_hidden: hidden units per direction in each BiLSTM layer
    """

    def __init__(
        self,
        num_classes: int,
        img_height: int = 32,
        lstm_hidden: int = 256,
    ):
        super().__init__()
        assert img_height % 16 == 0, "img_height must be divisible by 16"

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # H/2, W/2

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # H/4, W/4

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),               # H/8, W/4

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),               # H/16, W/4

            # Block 7 — collapses height to 1 when H=32
            nn.Conv2d(512, 512, kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, lstm_hidden, lstm_hidden),
            BidirectionalLSTM(lstm_hidden, lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W)
        features = self.cnn(x)             # (N, 512, 1, W')
        features = features.squeeze(2)     # (N, 512, W')
        features = features.permute(2, 0, 1)  # (W', N, 512) = (T, N, C)
        logits = self.rnn(features)        # (T, N, num_classes)
        return logits
