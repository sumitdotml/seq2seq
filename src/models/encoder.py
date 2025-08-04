import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        # shape: [src_vocab_size, embed_size]
        self.embedding = nn.Embedding(
            num_embeddings=src_vocab_size, embedding_dim=embed_size
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # here, x has to be [batch_size, seq_len] to begin with

        # [batch_size, seq_len] -> [batch_size, seq_len, embed_size]
        x_embedded = self.dropout(self.embedding(x))

        # the torch LSTM outputs out, (hidden, cell). we don't need the
        # `out` for Encoder since this is seq2seq
        _, (hidden_state, cell_state) = self.lstm(x_embedded)

        # hidden_state: [num_layers, batch_size, hidden_size]
        # cell_state: [num_layers, batch_size, hidden_size]
        return hidden_state, cell_state


if __name__ == "__main__":
    encoder = Encoder(
        src_vocab_size=10, embed_size=5, hidden_size=3, num_layers=10, dropout=0.1
    )
    x = torch.randint(0, 10, (2, 5))
    hidden_state, cell_state = encoder(x)
    print(f"hidden_state.shape: {hidden_state.shape}")
    print(f"cell_state.shape: {cell_state.shape}")
