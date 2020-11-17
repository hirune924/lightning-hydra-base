import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

from model.model import get_model
from dataset.dataset import get_dataset
from system.system import LitClassifier

# for temporary
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

@hydra.main(config_path='config', config_name="config.yaml")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # set seed
    seed_everything(2020)

    # set logger
    tb_logger = loggers.TensorBoardLogger(**cfg.logging.tb_logger)
    # set callback

    # set data
    #dataset = CIFAR10(to_absolute_path('data'), train=True, download=True, transform=transforms.ToTensor())
    #mnist_test = CIFAR10(to_absolute_path('data'), train=False, download=True, transform=transforms.ToTensor())
    #mnist_train, mnist_val = random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
    dataset = get_dataset(cfg.dataset)
    train_dataset = dataset['train']
    valid_dataset = dataset['valid']

    train_loader = DataLoader(train_dataset, **cfg.dataset.dataloader.train)
    val_loader = DataLoader(valid_dataset, **cfg.dataset.dataloader.valid)
    #test_loader = DataLoader(mnist_test, batch_size=32)
    
    # set model
    model = get_model(cfg.model)

    # set lit system
    lit_model = LitClassifier(hparams=cfg, model=model)

    # set trainer
    trainer = Trainer(
        logger=[tb_logger],
        **cfg.trainer
    )
    # training
    trainer.fit(lit_model, train_loader, val_loader)

    # test (if you need)
    #result = trainer.test(test_dataloaders=test_loader)
    #print(result)

if __name__ == "__main__":
    main()