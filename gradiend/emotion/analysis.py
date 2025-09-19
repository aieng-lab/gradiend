
data_path = 'data/emotion/AffectNet/train'
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import glob

class AffectNetNPYDataset(Dataset):
    def __init__(self, image_dir, annotations_dir, transform=None, task="expression", image_mode="disk"):
        """
        Args:
            image_dir (str): Path to images.
            annotations_dir (str): Path to annotation .npy files.
            transform (callable, optional): Transform to apply to images.
            task (str): One of ["expression", "valence_arousal", "all"].
            image_mode (str): One of ["disk", "preload", "cache"].
        """
        assert image_mode in ("disk", "preload", "cache"), "image_mode must be 'disk', 'preload', or 'cache'"

        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.task = task
        self.image_mode = image_mode

        self.image_map = {}   # id → image (if preloaded or cached) or path (if disk)
        self.annotations = {} # id → labels
        self.cache = {}       # for 'cache' mode

        self._collect_images()
        self._load_annotations()
        self.image_ids = sorted(self.annotations.keys())

    def _collect_images(self):
        for fname in sorted(os.listdir(self.image_dir)):
            if not fname.endswith(".jpg"):
                continue
            try:
                img_id = int(os.path.splitext(fname)[0])
                path = os.path.join(self.image_dir, fname)
                self.image_map[img_id] = path
            except ValueError:
                raise ValueError(f"Filename not convertible to integer ID: {fname}")

        if not self.image_map:
            raise RuntimeError("No valid .jpg images found in image_dir")

        if self.image_mode == "preload":
            for img_id, path in self.image_map.items():
                self.image_map[img_id] = self._load_image(path)

    def _load_annotations(self):
        ind_files = sorted(glob.glob(os.path.join(self.annotations_dir, "*_Ind.npy")))
        if not ind_files:
            raise FileNotFoundError("No *_Ind.npy files found.")

        valid_ids = set(self.image_map.keys())

        for ind_file in ind_files:
            prefix = os.path.basename(ind_file).split("_")[0]
            exp_file = os.path.join(self.annotations_dir, f"{prefix}_exp.npy")
            val_file = os.path.join(self.annotations_dir, f"{prefix}_val.npy")
            aro_file = os.path.join(self.annotations_dir, f"{prefix}_aro.npy")

            if not all(os.path.exists(f) for f in [exp_file, val_file, aro_file]):
                raise FileNotFoundError(f"Missing label files for prefix {prefix}")

            inds = np.load(ind_file).astype(int)
            exps = np.load(exp_file).astype(int)
            vals = np.load(val_file).astype(float)
            aros = np.load(aro_file).astype(float)

            for i, img_id in enumerate(inds):
                if img_id in valid_ids:
                    self.annotations[img_id] = {
                        "expression": exps[i],
                        "valence": vals[i],
                        "arousal": aros[i],
                    }

        missing_ids = valid_ids - self.annotations.keys()
        if missing_ids:
            raise ValueError(f"Missing annotations for image IDs: {sorted(missing_ids)[:5]} ...")

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # Image loading based on mode
        if self.image_mode == "preload":
            image = self.image_map[img_id]
        elif self.image_mode == "disk":
            path = self.image_map[img_id]
            image = self._load_image(path)
        elif self.image_mode == "cache":
            if img_id in self.cache:
                image = self.cache[img_id]
            else:
                path = self.image_map[img_id]
                image = self._load_image(path)
                self.cache[img_id] = image
        else:
            raise ValueError(f"Unknown image_mode: {self.image_mode}")

        ann = self.annotations[img_id]
        exp = ann["expression"]
        val = ann["valence"]
        aro = ann["arousal"]

        if self.task == "expression":
            return image, exp
        elif self.task == "valence_arousal":
            return image, torch.tensor([val, aro], dtype=torch.float32)
        elif self.task == "all":
            return image, exp, torch.tensor([val, aro], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown task: {self.task}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = AffectNetNPYDataset(
    image_dir=f"{data_path}/images",
    annotations_dir=f"{data_path}/annotations",
    transform=transform,
    task="all"
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


