import data
index1 = data.index1
index2 = data.index2

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer, LightningModule
df = index1["FP.CPI.TOTL.ZG"][["YR", "VALUE"]].rename(columns={"VALUE": "cpi"})
df = df.merge(index1["FR.INR.RINR"][["YR", "VALUE"]].rename(columns={"VALUE": "real_interest"}), on="YR", how="inner")
df = df.merge(index1["FR.INR.LEND"][["YR", "VALUE"]].rename(columns={"VALUE": "export"}), on="YR", how="inner")
df = df.merge(index1["GC.TAX.TOTL.GD.ZS"][["YR", "VALUE"]].rename(columns={"VALUE": "tax_ratio"}), on="YR", how="inner")
df = df.merge(index1["NE.EXP.GNFS.CD"][["YR", "VALUE"]].rename(columns={"VALUE": "export"}), on="YR", how="inner")
df = df.merge(index1["NE.IMP.GNFS.CD"][["YR", "VALUE"]].rename(columns={"VALUE": "export"}), on="YR", how="inner")

df.rename(columns={"YR": "time_idx"}, inplace=True)
df["group"] = "KOR"
df = df.dropna()

df["time_idx"] = df["time_idx"].astype("int")
df["cpi"] = df["cpi"].astype("float")

max_encoder_length = 5
max_prediction_length = 2

dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="cpi",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["cpi", "real_interest", "tax_ratio", "export"],
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

from pytorch_forecasting.metrics import QuantileLoss

tft = TemporalFusionTransformer.from_dataset(
    dataset,
    loss=QuantileLoss(),
    logging_metrics=[],
    learning_rate=0.03,
    hidden_size=8,
    attention_head_size=1,
    dropout=0.1,
    log_interval=-1,
    reduce_on_plateau_patience=4
)

from pytorch_lightning import Trainer, LightningModule

# Wrap TemporalFusionTransformer with a LightningModule
class WrappedTFT(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.model.loss(output, batch["target"])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)

wrapped_tft = WrappedTFT(tft)

trainer = Trainer(
    max_epochs=10,
    gradient_clip_val=0.1,
    enable_model_summary=True,
    accelerator="cpu",
    devices=1,
)

print(type(tft))
trainer.fit(wrapped_tft, train_dataloaders=dataloader)