import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MouseDataset(Dataset):
    def __init__(self, jsonl_file):
        self.samples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                dx = record["relative_move"]["dx"]
                dy = record["relative_move"]["dy"]
                traj = record["trajectory"]  # list [x,y], len=10
                traj = np.array(traj, dtype=np.float32).flatten()  # 10*2=20
                self.samples.append(((dx, dy), traj))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (dx, dy), traj = self.samples[idx]
        x = np.array([dx, dy], dtype=np.float32)
        y = traj
        return x, y

class TrajNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )

    def forward(self, x):
        return self.net(x)

def train_model(jsonl_file, save_path="mouse_traj.onnx", epochs=50, batch_size=32, lr=0.001):
    dataset = MouseDataset(jsonl_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TrajNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")

    dummy_input = torch.randn(1, 2)
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=["input"], output_names=["trajectory"],
        dynamic_axes={"input": {0: "batch"}, "trajectory": {0: "batch"}},
        opset_version=11
    )
    print(f"模型已保存為 {save_path}")

if __name__ == "__main__":
    train_model("mouse_dataset.jsonl")
