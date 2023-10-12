from pathlib import Path
import torch
from torch.utils.data import Dataset
import polars as pl

BASE_PATH = Path("/home/infres/jma-21/J2Letters_RNSA_2023/rsna-2023-abdominal-trauma-detection")
PT_PATH = Path("/home/infres/jma-21/J2Letters_RNSA_2023/rnsa-2023-5mm-slices-pt")

LABEL_COLS = ["bowel_healthy", "bowel_injury", "extravasation_healthy", "extravasation_injury", "kidney_healthy", "kidney_low", "kidney_high", "liver_healthy", "liver_low", "liver_high", "spleen_healthy", "spleen_low", "spleen_high", "any_injury"]
LABEL_MODEL_COLS = ["bowel_healthy", "extravasation_healthy", "kidney_healthy", "kidney_low", "kidney_high", "liver_healthy", "liver_low", "liver_high", "spleen_healthy", "spleen_low", "spleen_high"]

class RNSA2023Dataset(Dataset):
    def __init__(self, labels_data, idx):
        self.labels = labels_data[idx]
        self.patients_series = pl.read_csv(BASE_PATH.joinpath("train_series_meta.csv")).join(self.labels.select("patient_id"), on="patient_id", how="inner")
        
    def __len__(self):
        return len(self.patients_series)
    
    def __getitem__(self, idx):
        patient, serie, _, _ = self.patients_series[idx]
        data = torch.load(PT_PATH.joinpath(str(patient[0])).joinpath(f"{str(serie[0])}.pt"))#, map_location=torch.device(device)) # Incompatible with Lightning
        # Do not consider any_injury from LABEL_COLS as a label
        label = torch.from_numpy(self.labels.filter(pl.col("patient_id")==patient[0]).select(pl.col(LABEL_MODEL_COLS)).to_numpy())#.to(device) # Incompatible with Lightning
        return data, label
    