import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import TrafficDataset
from model import TrafficTransformer
from utils import train_epoch
from config import Config


cfg = Config()
device = torch.device("cpu")

print("Loading dataset...")

dataset = TrafficDataset(
    cfg.DATA_PATH,
    cfg.SEQ_LEN,
    cfg.PRED_LEN,
    cfg.MAX_SAMPLES
)

loader = DataLoader(
    dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

input_dim = dataset.data.shape[1]

model = TrafficTransformer(
    input_dim,
    cfg.D_MODEL,
    cfg.N_HEADS,
    cfg.NUM_LAYERS,
    cfg.PRED_LEN
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

print("Training...\n")

for epoch in range(cfg.EPOCHS):

    loss = train_epoch(model, loader, optimizer, criterion, device)

    print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Loss: {loss:.4f}")

torch.save(model.state_dict(), cfg.MODEL_PATH)

print("\nModel saved")