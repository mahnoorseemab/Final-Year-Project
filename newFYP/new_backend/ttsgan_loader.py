# ============================================================
# TTSGAN_LOADER.PY — UPDATED FOR TTSWGAN___DCTGAN_FINAL.ipynb
# (8-pair-type, single-scalar-feature design)
#
# CHANGE — N_FEATURES: 5 -> 1
#   FINAL notebook Chunk 9 prints: NUM_FEATURES = 1
#   Confirmed from Chunk 6 sequence-creation code: every sequence
#   reshapes to (entities, seq_len, 1) -- one scalar per timestep,
#   not 5 stacked ID columns.
#
#   input_projection in discriminator: Linear(1, 64) not Linear(5, 64)
#   output_projection in generator:    Linear(64, 1) not Linear(64, 5)
#
# NOTE ON TRAINING HYPERPARAMETER:
#   This model was trained with TTS_LAMBDA_GP = 10 (DCT uses 1 --
#   see dctgan_loader.py). This does not change the saved weight
#   shapes, only training behaviour; no loader code change needed,
#   included here for documentation only.
#
# UNCHANGED:
#   SEQ_LEN = 10
#   N_LAYERS = 3 (TTS uses 3, DCT uses 2)
#   D_MODEL = 64, N_HEADS = 4, DIM_FF = 128, DROPOUT = 0.1
#   Sinusoidal positional encoding in generator (register_buffer)
#   Learnable positional embedding in discriminator (nn.Parameter)
#   No Sigmoid on discriminator -- still correct for WGAN-GP
# ============================================================

import torch
import numpy as np
import torch.nn as nn
import os

device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ttsgan_model = None

# ── Settings — exact match to TTSWGAN___DCTGAN_FINAL.ipynb Chunk 9 ──
SEQ_LEN    = 10    # unchanged
N_FEATURES = 1     # CHANGED: 5 -> 1 (one scalar per timestep)
LATENT_DIM = 64
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 3     # unchanged
DIM_FF     = 128
DROPOUT    = 0.1


# ── Sinusoidal positional encoding (matches training) ─────────
def get_sinusoidal_encoding(seq_len, d_model):
    pe = torch.zeros(1, seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
    )
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


# ── Generator — N_FEATURES updated to 1 ───────────────────────
class TTSGANGenerator(nn.Module):
    def __init__(self):
        super(TTSGANGenerator, self).__init__()
        self.input_projection = nn.Linear(LATENT_DIM, D_MODEL)
        # Sinusoidal — NOT nn.Parameter (matches training)
        self.register_buffer(
            'pos_embedding',
            get_sinusoidal_encoding(SEQ_LEN, D_MODEL)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=DIM_FF, dropout=DROPOUT, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.output_projection = nn.Sequential(
            nn.Linear(D_MODEL, N_FEATURES),
            nn.Sigmoid()
        )

    def forward(self, noise):
        # noise is (batch, SEQ_LEN, LATENT_DIM) -- per timestep
        x = self.input_projection(noise)   # (batch, 10, 64)
        x = x + self.pos_embedding         # add sinusoidal position
        x = self.transformer(x)
        return self.output_projection(x)


# ── Discriminator — N_FEATURES updated to 1 ───────────────────
class TTSGANDiscriminator(nn.Module):
    def __init__(self):
        super(TTSGANDiscriminator, self).__init__()

        # Input projection: 1 feature -> 64 (CHANGED from 5 -> 64)
        self.input_projection = nn.Linear(N_FEATURES, D_MODEL)

        # pos_embedding shape: (1, 10, 64)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, SEQ_LEN, D_MODEL)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = D_MODEL,
            nhead           = N_HEADS,
            dim_feedforward = DIM_FF,
            dropout         = DROPOUT,
            batch_first     = True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=N_LAYERS
        )

        # NO Sigmoid — WGAN-GP requires raw unbounded scores
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)           # average-pool all 10 timesteps
        return self.classifier(x)


# ── Load ─────────────────────────────────────────────────────
def load_ttsgan():
    global ttsgan_model

    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    gen_path  = os.path.join(MODELS_DIR, 'ttsgan_generator.pth')
    disc_path = os.path.join(MODELS_DIR, 'ttsgan_discriminator.pth')

    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"Missing: {gen_path}")
    if not os.path.exists(disc_path):
        raise FileNotFoundError(f"Missing: {disc_path}")

    gen = TTSGANGenerator().to(device)
    dis = TTSGANDiscriminator().to(device)

    gen.load_state_dict(torch.load(gen_path,  map_location=device))
    dis.load_state_dict(torch.load(disc_path, map_location=device))

    gen.eval()
    dis.eval()

    ttsgan_model = {'generator': gen, 'discriminator': dis}
    print("TTSGAN loaded!")
    print(f"   SEQ_LEN    : {SEQ_LEN}")
    print(f"   N_FEATURES : {N_FEATURES}  (single scalar per timestep -- 8-pair-type design)")
    print(f"   N_LAYERS   : {N_LAYERS}")
    print(f"   Score type : raw WGAN-GP (unbounded, no Sigmoid)")
    return ttsgan_model
