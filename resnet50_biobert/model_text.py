"""
src/model_text.py

BioBERT + BiLSTM => (B, l_q, 512)
Analogie zum Paper: "Question Feature Extraction"
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class BioBERTBiLSTM(nn.Module):
    def __init__(self, biobert_model="dmis-lab/biobert-v1.1", lstm_hidden=256,
                 dropout=0.1, max_len=40):
        super().__init__()
        self.biobert = AutoModel.from_pretrained(biobert_model)
        hidden_size = self.biobert.config.hidden_size  # 768

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.output_dim = lstm_hidden * 2  # 512
        self.linear = nn.Linear(self.output_dim, 512)
        self.dropout = nn.Dropout(dropout)

        from resnet50_biobert.model_transformer import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model=512, max_len=max_len)

    def forward(self, input_ids, attention_mask):
        """
        input_ids, attention_mask: (B, seq_len)
        return: (B, seq_len, 512)
        """
        outputs = self.biobert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, seq_len, 768)
        lstm_out, _ = self.bilstm(last_hidden)   # (B, seq_len, 512)
        proj = self.linear(lstm_out)
        proj = self.dropout(proj)
        proj = proj.transpose(0, 1)
        proj = self.pos_encoding(proj)
        proj = proj.transpose(0, 1)
        return proj
