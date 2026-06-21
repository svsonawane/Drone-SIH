import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
import torch, torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR  = "images"
MASK_DIR = "masks"
IMG_SIZE = 512
BATCH    = 2
EPOCHS   = 20
GSD      = 0.25

CLASS_NAMES  = ["Background", "Building", "Woodland", "Water", "Road"]
CLASS_COLORS = np.array([
    [0,   0,   0  ],
    [255, 0,   0  ],
    [0,   255, 0  ],
    [0,   0,   255],
    [128, 128, 128],
], dtype=np.uint8)

# ── Build image ID list ───────────────────────────────────────────────────────
ids = []
for f in os.listdir(IMG_DIR):
    if f.lower().endswith((".jpg", ".png", ".tif")):
        ids.append(os.path.splitext(f)[0])

print(f"Total images found: {len(ids)}")
if len(ids) == 0:
    raise FileNotFoundError(f"No images found in '{IMG_DIR}'. Check your folder path.")

X_tv,    X_test = train_test_split(ids, test_size=0.1,  random_state=42)
X_train, X_val  = train_test_split(X_tv, test_size=0.15, random_state=42)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
def find_file(folder, name):
    for ext in [".png", ".jpg", ".tif"]:
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {name} in {folder}")

class LandDataset(Dataset):
    def __init__(self, ids, transform=None):
        self.ids = ids
        self.transform = transform
        self.norm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        img  = cv2.cvtColor(cv2.imread(find_file(IMG_DIR,  self.ids[i])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(find_file(MASK_DIR, self.ids[i]), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        img  = self.norm(Image.fromarray(img))
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask

train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])
val_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

train_loader = DataLoader(LandDataset(X_train, train_tf), batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(LandDataset(X_val,   val_tf),   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(LandDataset(X_test,  val_tf),   batch_size=1,     shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
model     = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=5).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Training ──────────────────────────────────────────────────────────────────
best_val = float("inf")
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        loss = criterion(model(imgs), masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        t_loss += loss.item()
    t_loss /= len(train_loader)

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            v_loss += criterion(model(imgs.to(DEVICE)), masks.to(DEVICE)).item()
    v_loss /= len(val_loader)

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    print(f"  Train Loss: {t_loss:.4f}  |  Val Loss: {v_loss:.4f}")

    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), "models/best.pth")
        print("  💾 Model saved")

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
plt.savefig("outputs/loss.png"); plt.show()
print("✅ Training done! Loss curve saved to outputs/loss.png")

# ── Predict + Area Report ─────────────────────────────────────────────────────
model.load_state_dict(torch.load("models/best.pth", map_location=DEVICE))
model.eval()

for i, (imgs, _) in enumerate(test_loader):
    with torch.no_grad():
        pred = model(imgs.to(DEVICE)).squeeze(0).argmax(0).cpu().numpy()

    orig = imgs.squeeze(0).permute(1, 2, 0).numpy()
    orig = (orig * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(orig);                ax1.set_title("Satellite Image"); ax1.axis("off")
    ax2.imshow(CLASS_COLORS[pred]);  ax2.set_title("Predicted Mask");  ax2.axis("off")

    patches = [mpatches.Patch(color=CLASS_COLORS[j]/255, label=CLASS_NAMES[j]) for j in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.tight_layout()
    plt.savefig(f"outputs/pred_{i}.png", bbox_inches="tight")
    plt.show()

    total_px = pred.size
    print(f"\n📐 Area Report — Image {i}")
    print(f"{'Class':<14} {'Pixels':>8} {'Area (m²)':>12} {'Area (ha)':>10} {'% of image':>10}")
    print("-" * 58)
    for cls_i, name in enumerate(CLASS_NAMES):
        px = int((pred == cls_i).sum())
        if px > 0:
            m2  = px * GSD ** 2
            pct = px / total_px * 100
            print(f"{name:<14} {px:>8,} {m2:>12,.1f} {m2/10000:>10.4f} {pct:>9.1f}%")
    print("-" * 58)

    if i >= 4:
        break

print("\n✅ All predictions saved to outputs/")