import pytorch_lightning as pl
from lightning_modules.autoencoder import AutoEncoder
from utils.misc import load_configs

CONFIG_FILE = "configs/config.yaml"

if __name__ == '__main__':
    configs = load_configs(CONFIG_FILE)
    model = AutoEncoder(
        hparams=configs
    )
    trainer = pl.Trainer(gpus=1, max_epochs=configs["max_epochs"])
    trainer.fit(model)
    trainer.test()