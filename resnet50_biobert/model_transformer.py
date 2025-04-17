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

# Wrapper-Klassen, falls benötigt (hier als Platzhalter)
class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """ Wrapper für PyTorch TransformerEncoderLayer (batch_first=True) """
    pass

class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    """ Wrapper für PyTorch TransformerDecoderLayer (batch_first=True) """
    pass

class TraPVQA(nn.Module):
    """
    Gesamtmodell:
      - Text: BioBERT+BiLSTM -> (B, lQ, 512)
      - Bild: ResNet50 -> (B, 49, 512)
      - Fusion der Features mittels zweistufiger Multihead-Attention (statt einfacher Konkatenation)
      - Transformer Decoder (2-Layer) zur Generierung der Antwort
    """
    def __init__(self, text_encoder, image_encoder,
                 vocab_size=30522,
                 nhead=8,
                 num_encoder_layers=2,  # Nicht mehr genutzt, da Fusion manuell implementiert wird
                 num_decoder_layers=2,
                 dim_feedforward=2048,
                 dropout=0.1,
                 eos_token_id=3):  # Setze EOS-Token ID entsprechend deines Tokenizers
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        # Fusion: zweistufige Multihead-Attention Module
        self.fusion_mha1 = nn.MultiheadAttention(embed_dim=512, num_heads=nhead, dropout=dropout, batch_first=True)
        self.fusion_norm1 = nn.LayerNorm(512)
        self.fusion_ffn1 = nn.Sequential(
            nn.Linear(512, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 512)
        )
        self.fusion_norm2 = nn.LayerNorm(512)

        self.fusion_mha2 = nn.MultiheadAttention(embed_dim=512, num_heads=nhead, dropout=dropout, batch_first=True)
        self.fusion_norm3 = nn.LayerNorm(512)
        self.fusion_ffn2 = nn.Sequential(
            nn.Linear(512, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 512)
        )
        self.fusion_norm4 = nn.LayerNorm(512)

        # Transformer Decoder
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

        self.eos_token_id = eos_token_id  # EOS-Token zum Beenden der Inferenz

    def forward(self, images, input_ids, attention_mask, decoder_input_ids=None):
        """
        images: (B, 3, 224, 224)
        input_ids, attention_mask: für den Textencoder (B, lQ)
        decoder_input_ids: (B, lAns) -> Teacher Forcing, wenn vorhanden
        """
        # 1) Frage-Features: (B, lQ, 512)
        xq = self.text_encoder(input_ids, attention_mask)
        # 2) Bild-Features: (B, 49, 512)
        xi = self.image_encoder(images)

        # 3) Fusion der Features in zwei Schritten:
        # Erster Schritt: Query und Key aus Fragefeatures, Value sind Bildfeatures
        attn_output1, _ = self.fusion_mha1(query=xq, key=xq, value=xi)
        fusion1 = self.fusion_norm1(xq + attn_output1)
        ffn_output1 = self.fusion_ffn1(fusion1)
        fusion1 = self.fusion_norm2(fusion1 + ffn_output1)
        # Zweiter Schritt: Query und Key aus fusion1, Value bleiben Bildfeatures
        attn_output2, _ = self.fusion_mha2(query=fusion1, key=fusion1, value=xi)
        fusion2 = self.fusion_norm3(fusion1 + attn_output2)
        ffn_output2 = self.fusion_ffn2(fusion2)
        enc_out = self.fusion_norm4(fusion2 + ffn_output2)
        # enc_out hat nun die Form (B, lQ, 512)

        if decoder_input_ids is not None:
            # Teacher Forcing-Modus
            tgt_emb = self.decoder_embedding(decoder_input_ids)  # (B, lAns, 512)
            tgt_emb = tgt_emb.transpose(0, 1)                    # (lAns, B, 512)
            tgt_emb = self.pos_encoding_dec(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)                    # (B, lAns, 512)
            lAns = decoder_input_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(lAns).to(tgt_emb.device)
            dec_out = self.transformer_decoder(
                tgt=tgt_emb,
                memory=enc_out,
                tgt_mask=tgt_mask
            )
            logits = self.output_layer(dec_out)  # (B, lAns, vocab_size)
            return logits
        else:
            # Inferenzmodus (autoregressiv, dynamischer Abbruch bei Erreichen des EOS)
            max_len = 20
            batch_size = images.size(0)
            # Starte mit <SOS>; hier hartkodiert als Token-ID 2 (anpassen, falls nötig)
            dec_input = torch.full((batch_size, 1), 2, dtype=torch.long, device=images.device)
            outputs = []
            for t in range(max_len):
                tgt_emb = self.decoder_embedding(dec_input)
                tgt_emb = tgt_emb.transpose(0, 1)
                tgt_emb = self.pos_encoding_dec(tgt_emb)
                tgt_emb = tgt_emb.transpose(0, 1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(1)).to(dec_input.device)
                dec_out = self.transformer_decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)
                logit_t = self.output_layer(dec_out[:, -1, :])  # (B, vocab_size)
                next_token = logit_t.argmax(dim=1, keepdim=True)   # (B, 1)
                outputs.append(next_token)
                dec_input = torch.cat([dec_input, next_token], dim=1)
                # Prüfe, ob in allen Batch-Elementen der EOS-Token erschienen ist
                if (next_token == self.eos_token_id).all():
                    break
            outputs = torch.cat(outputs, dim=1)  # (B, generierte Sequenzlänge)
            return outputs
