from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb
from pytorch_lightning.loggers import WandbLogger


if __name__ == "__main__":
    wandb.init(project="math-offline-fptu", name="BTTR")
    model = LitBTTR(d_model=256, growth_rate=24, num_layers=16, nhead=8, num_decoder_layers=3, dim_feedforward=1024, dropout=0.3, beam_size=10, max_len=200, alpha=1.0, learning_rate=1.0, patience=20)
    dm = CROHMEDatamodule(zipfile_path="data.zip", batch_size=32, num_workers=15, test_year="2014")

    trainer = Trainer(
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(filename='{epoch}-{step}-{val_ExpRate:.4f}', save_top_k=5, monitor='val_ExpRate', mode='max'),
        ], 
        check_val_every_n_epoch=5,
        fast_dev_run=True,
        deterministic=False, 
        max_epochs=200, 
        accelerator='gpu',
        devices=1,
        logger=WandbLogger(),
    )

    trainer.fit(model, dm)
    wandb.finish()