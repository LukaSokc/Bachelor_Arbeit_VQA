"""
src/model_transformer.py

Transformer-Encoder/Decoder für TraP-VQA
(2 Encoder-Layer, 2 Decoder-Layer, etc.)
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (seq_len, B, d_model)
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :].unsqueeze(1)
        return x


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """ Wrapper für PyTorch TransformerEncoderLayer (batch_first=True) """
    pass

class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    """ Wrapper für PyTorch TransformerDecoderLayer (batch_first=True) """
    pass


class TraPVQA(nn.Module):
    """
    Gesamtmodell:
      - BioBERT+BiLSTM (Text) -> (B, lQ, 512)
      - ResNet50 (Image) -> (B, 49, 512)
      - 2-Layer TransformerEncoder (fuse text+image)
      - 2-Layer TransformerDecoder (Antwort generieren)
    """
    def __init__(self, text_encoder, image_encoder,
                 vocab_size=30522,
                 nhead=8,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        encoder_layer = TransformerEncoderLayer(
            d_model=512, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = TransformerDecoderLayer(
            d_model=512, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.decoder_embedding = nn.Embedding(vocab_size, 512)
        self.pos_encoding_dec = PositionalEncoding(d_model=512, max_len=50)

        self.output_layer = nn.Linear(512, vocab_size)

    def forward(self, images, input_ids, attention_mask, decoder_input_ids=None):
        # 1) Textfeatures
        xq = self.text_encoder(input_ids, attention_mask)   # (B, lQ, 512)
        # 2) Bildfeatures
        xi = self.image_encoder(images)                     # (B, 49, 512)

        # 3) Transformer Encoder
        combined = torch.cat([xq, xi], dim=1)  # (B, lQ+49, 512)
        enc_out = self.transformer_encoder(combined)

        # 4) Transformer Decoder
        if self.training and decoder_input_ids is not None:
            # Teacher Forcing
            tgt_emb = self.decoder_embedding(decoder_input_ids)  # (B, lAns, 512)
            tgt_emb = tgt_emb.transpose(0, 1)  # => (lAns, B, 512)
            tgt_emb = self.pos_encoding_dec(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)  # => (B, lAns, 512)

            lAns = decoder_input_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(lAns).to(tgt_emb.device)

            dec_out = self.transformer_decoder(
                tgt=tgt_emb,
                memory=enc_out,
                tgt_mask=tgt_mask
            )  # (B, lAns, 512)

            logits = self.output_layer(dec_out)  # (B, lAns, vocab_size)
            return logits
        else:
            # Inference (Greedy)
            max_len = 20
            batch_size = images.size(0)
            dec_input = torch.full((batch_size, 1), 2, dtype=torch.long, device=images.device)  # <SOS>=2
            outputs = []

            for t in range(max_len):
                tgt_emb = self.decoder_embedding(dec_input)
                tgt_emb = tgt_emb.transpose(0, 1)
                tgt_emb = self.pos_encoding_dec(tgt_emb)
                tgt_emb = tgt_emb.transpose(0, 1)

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(1)).to(dec_input.device)
                dec_out = self.transformer_decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)
                logit_t = self.output_layer(dec_out[:, -1, :])  # (B, vocab_size)
                next_token = logit_t.argmax(dim=1, keepdim=True)
                outputs.append(next_token)
                dec_input = torch.cat([dec_input, next_token], dim=1)

            outputs = torch.cat(outputs, dim=1)  # (B, max_len)
            return outputs
