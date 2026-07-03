# ============================================================
# DCTGAN_LOADER.PY — UPDATED FOR TTSWGAN___DCTGAN_FINAL.ipynb
# (8-pair-type, single-scalar-feature design)
#
# CHANGES FROM PREVIOUS VERSION (5-feature design):
#
# CHANGE 1 — N_FEATURES: 5 → 1
#   FINAL notebook Chunk 18 prints: DCT_N_FEATURES = 1
#   Every sequence is ONE scalar value per timestep (e.g. the
#   SERVICE_ID of the doctor's last 10 transactions), NOT 5
#   stacked ID columns. Confirmed directly from the notebook's
#   Chunk 6 sequence-creation code: sequences reshape to
#   (entities, seq_len, 1).
#
#   input_projection in discriminator: Linear(1, 64) not Linear(5, 64)
#   output_projection in generator:    Linear(64, 1) not Linear(64, 5)
#
# CHANGE 2 — Trained with DCT_LAMBDA_GP = 1 (TTS uses 10 — see
#   ttsgan_loader.py). This does not affect the saved weight
#   shapes, only how they were trained; no loader code change
#   needed for this, included here for documentation only.
#
# CHANGE 3 — SEQ_LEN stays 10 (unchanged from before).
#
# UNCHANGED:
#   N_LAYERS = 2       — DCT still uses 2 Transformer layers
#   D_MODEL = D_CHANNELS = 64 — unchanged
#   N_HEADS = 4        — unchanged
#   DilatedCNNBlock architecture — unchanged
#   No Sigmoid on discriminator — still correct for WGAN-GP
#   Sigmoid on generator output — still correct
#   Generator filename: dctgan_generator.pth (no suffix)
# ============================================================

import torch
import torch.nn as nn
import os

device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dctgan_model = None

# ── Settings — exact match to TTSWGAN___DCTGAN_FINAL.ipynb Chunk 18 ──
SEQ_LEN      = 10   # unchanged
N_FEATURES   = 1    # CHANGED: 5 -> 1 (one scalar per timestep)
LATENT_DIM   = 64
D_CHANNELS   = 64
D_MODEL      = 64
N_HEADS      = 4
N_LAYERS     = 2    # DCT uses 2 layers (TTS uses 3)
KERNEL_SIZE  = 3


# ── DilatedCNNBlock — unchanged ───────────────────────────────
class DilatedCNNBlock(nn.Module):
    def __init__(self):
        super(DilatedCNNBlock, self).__init__()
        self.conv_d1 = nn.Conv1d(D_CHANNELS, D_CHANNELS, KERNEL_SIZE, dilation=1, padding=1)
        self.conv_d2 = nn.Conv1d(D_CHANNELS, D_CHANNELS, KERNEL_SIZE, dilation=2, padding=2)
        self.conv_d4 = nn.Conv1d(D_CHANNELS, D_CHANNELS, KERNEL_SIZE, dilation=4, padding=4)
        self.combine = nn.Conv1d(D_CHANNELS * 3, D_CHANNELS, kernel_size=1)
        self.norm    = nn.LayerNorm(D_CHANNELS)
        self.relu    = nn.ReLU()

    def forward(self, x):
        out_d1 = self.relu(self.conv_d1(x))
        out_d2 = self.relu(self.conv_d2(x))
        out_d4 = self.relu(self.conv_d4(x))
        out    = torch.cat([out_d1, out_d2, out_d4], dim=1)
        out    = self.combine(out)
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


# ── Generator — N_FEATURES updated to 1 ───────────────────────
class DCTGANGenerator(nn.Module):
    def __init__(self):
        super(DCTGANGenerator, self).__init__()
        self.input_projection  = nn.Linear(LATENT_DIM, D_CHANNELS)

        # pos_embedding: (1, 10, 64)
        self.pos_embedding     = nn.Parameter(torch.randn(1, SEQ_LEN, D_CHANNELS))
        self.dilated_cnn       = DilatedCNNBlock()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # Output: 64 -> 1 (CHANGED from 5), Sigmoid for [0,1]
        self.output_projection = nn.Sequential(
            nn.Linear(D_CHANNELS, N_FEATURES),
            nn.Sigmoid()
        )

    def forward(self, noise):
        x = self.input_projection(noise)
        x = x.unsqueeze(1).expand(-1, SEQ_LEN, -1)
        x = x + self.pos_embedding
        x = x.transpose(1, 2)       # -> (batch, channels, seq_len) for Conv1d
        x = self.dilated_cnn(x)
        x = x.transpose(1, 2)       # -> (batch, seq_len, channels) for Transformer
        x = self.transformer(x)
        return self.output_projection(x)


# ── Discriminator — N_FEATURES updated to 1 ───────────────────
class DCTGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCTGANDiscriminator, self).__init__()

        # Input: 1 feature -> 64 (CHANGED from 5 -> 64)
        self.input_projection = nn.Linear(N_FEATURES, D_MODEL)

        # pos_embedding: (1, 10, 64)
        self.pos_embedding = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

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
        x = x.mean(dim=1)
        return self.classifier(x)


# ── Load ──────────────────────────────────────────────────────
def load_dctgan():
    global dctgan_model

    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    gen_path  = os.path.join(MODELS_DIR, 'dctgan_generator.pth')
    disc_path = os.path.join(MODELS_DIR, 'dctgan_discriminator.pth')

    if not os.path.exists(gen_path):
        raise FileNotFoundError(
            f"Missing: {gen_path}\n"
            f"Expected 'dctgan_generator.pth' from TTSWGAN___DCTGAN_FINAL.ipynb."
        )
    if not os.path.exists(disc_path):
        raise FileNotFoundError(f"Missing: {disc_path}")

    generator = DCTGANGenerator().to(device)
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    generator.eval()

    dis = DCTGANDiscriminator().to(device)
    dis.load_state_dict(torch.load(disc_path, map_location=device))
    dis.eval()

    dctgan_model = {'generator': generator, 'discriminator': dis}
    print("DCTGAN loaded!")
    print(f"   SEQ_LEN    : {SEQ_LEN}")
    print(f"   N_FEATURES : {N_FEATURES}  (single scalar per timestep -- 8-pair-type design)")
    print(f"   N_LAYERS   : {N_LAYERS}")
    print(f"   Score type : raw WGAN-GP (unbounded, no Sigmoid)")
    print(f"   Gen file   : dctgan_generator.pth")
    return dctgan_model
