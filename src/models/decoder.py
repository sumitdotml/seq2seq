import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size, embedding_dim=embed_size
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(
            in_features=hidden_size, out_features=tgt_vocab_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor, cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # input shape: [batch_size] - contains ONE token per example in the batch
        # At each decoder step, each example in the batch gets exactly one input token:
        # - During training: the correct previous target token (teacher forcing)
        # - During inference: the token I predicted in the previous step
        # Example: if batch_size=3, input might be [9, 12, 5] meaning:
        #   - Example 1 gets token 9, Example 2 gets token 12, Example 3 gets token 5

        # [batch_size] -> [batch_size, 1]
        # adding sequence dimension since my LSTM expects [batch_size, seq_len] format
        # this makes it [batch_size, seq_len=1] because I process one token at a time
        input = input.unsqueeze(1)

        # [batch_size, 1] -> [batch_size, 1, embed_size]
        # convert token indices to dense embedding vectors
        # each of the batch_size tokens becomes an embed_size dimensional vector
        embedded_input = self.embedding(input)

        # output: [batch_size, 1, hidden_size]
        # processing through LSTM with previous hidden/cell states from encoder or previous step
        # the LSTM maintains separate hidden/cell states for each example in the batch
        output, (hidden, cell_state) = self.lstm(embedded_input, (hidden, cell_state))

        # [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        # removing the sequence dimension since I only had seq_len=1
        # now each example in the batch has a hidden_size dimensional representation
        output_squeezed = output.squeeze(1)

        # [batch_size, hidden_size] -> [batch_size, tgt_vocab_size]
        # projecting to vocabulary size to get logits for next token prediction
        # each example gets a score for every possible token in target vocabulary
        projection = self.output_projection(output_squeezed)

        return projection, hidden, cell_state
