import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from pytorch_lightning import Trainer


##############
# TRANSFORMS #
##############
class RotationalTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


class VerticalFlip:
    def __init__(self):
        pass

    def __call__(self, x):
        return TF.vflip(x)


class HorizontalFlip:
    def __init__(self):
        pass

    def __call__(self, x):
        return TF.hflip(x)


###########
# DATASET #
###########
class SelfSupervisedDataset(object):
    def __init__(self, image_path=Path("../data/imagenette2-320/train")):
        self.imgs = list(image_path.glob('**/*.JPEG'))
        self.class_transforms = [RotationalTransform(0), RotationalTransform(90),
                                 RotationalTransform(180), RotationalTransform(270),
                                 HorizontalFlip(), VerticalFlip()]
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.classes = len(self.class_transforms)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        label = random.choice(range(0, self.classes))
        img = img.convert("RGB")
        # Resize first, then apply our selected transform and finally convert to tensor
        transformed_image = self.to_tensor(self.class_transforms[label](transforms.Resize((224, 224))(img)))
        return transformed_image, label

    def __len__(self):
        return len(self.imgs)


#########
# MODEL #
#########
class SelfSupervisedModel(pl.LightningModule):

    def __init__(self, hparams=None, num_classes=6, batch_size=64, pretraining=True):
        super(SelfSupervisedModel, self).__init__()
        self.val_dataset = None
        self.training_dataset = None
        self.resnet = torchvision.models.resnet34(pretrained=False)
        self.resnet.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_classes))
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        if hparams is None:
            hparams = {}
        if "lr" not in hparams:
            hparams["lr"] = 0.001
        self.save_hyperparameters(hparams)
        self.valid_outputs = []
        self.pretraining = pretraining

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.loss_fn(predictions, targets)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def prepare_data(self):
        self.training_dataset = SelfSupervisedDataset()
        self.val_dataset = SelfSupervisedDataset(Path("imagenette2-320/val"))

    def train_dataloader(self):
        if self.pretraining:
            print('Training!')
            return torch.utils.data.DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=4,
                                               shuffle=True)
        else:
            print('Validation!')
            return imagenette_training_data_loader

    def val_dataloader(self):
        if self.pretraining:
            print('Training!')
            return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
        else:
            print('Validation!')
            return imagenette_val_data_loader

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        val_loss = self.loss_fn(predictions, targets)
        _, preds = torch.max(predictions, 1)
        # noinspection PyTypeChecker
        acc = torch.sum(preds == targets.data) / (targets.shape[0] * 1.0)
        self.valid_outputs.append({'val_loss': val_loss, 'val_acc': acc})  # Store outputs in an instance attribute
        return {'val_loss': val_loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.valid_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'].float() for x in self.valid_outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        self.valid_outputs = []  # Clear the outputs for the next epoch
        self.log_dict(logs)
        return {'progress_bar': logs}


########################
# TRAINING ON RAW DATA #            training was here on rotations, flipping etc., pretraining = True
########################
#model = SelfSupervisedModel({'lr': 0.001}, pretraining=True)
#trainer = pl.Trainer(max_epochs=5, accelerator="auto")
#trainer.fit(model)
#trainer.save_checkpoint("models/self_supervised_simple_labelling.pth")

###############################
# TRAINING ON SUPERVISED DATA #     training was here to determine the class from the dataset, pretraining = False
###############################
#tfms = transforms.Compose([
#           transforms.Resize((224, 224)),
#           transforms.ToTensor()
#        ])
#imagenette_training_data = torchvision.datasets.ImageFolder(root="data/imagenette2-320/train/", transform=tfms)
#imagenette_training_data_loader = torch.utils.data.DataLoader(imagenette_training_data, batch_size=64, num_workers=4,
#                                                              shuffle=True)
#imagenette_val_data = torchvision.datasets.ImageFolder(root="data/imagenette2-320/val/", transform=tfms)
#imagenette_val_data_loader = torch.utils.data.DataLoader(imagenette_val_data, batch_size=64, num_workers=4)
#model = SelfSupervisedModel({'lr': 0.001}, pretraining=False)
#model = model.load_from_checkpoint("models/self_supervised_simple_labelling.pth")
#model.resnet.fc[2] = nn.Linear(256, 12)

#trainer = pl.Trainer(max_epochs=5, accelerator="auto")
#trainer.fit(model, train_dataloaders=imagenette_training_data_loader, val_dataloaders=imagenette_val_data_loader)


#########
# CHECK #   recreation of model from scratch, 10 epochs and non-supervised dataloaders
#########
tfms = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor()
        ])
imagenette_training_data = torchvision.datasets.ImageFolder(root="data/imagenette2-320/train/", transform=tfms)
imagenette_training_data_loader = torch.utils.data.DataLoader(imagenette_training_data, batch_size=64, num_workers=4,
                                                              shuffle=True)
imagenette_val_data = torchvision.datasets.ImageFolder(root="data/imagenette2-320/val/", transform=tfms)
imagenette_val_data_loader = torch.utils.data.DataLoader(imagenette_val_data, batch_size=64, num_workers=4)
standard_model = SelfSupervisedModel({'lr': 0.001}, pretraining=False)
trainer = pl.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(standard_model, train_dataloaders=imagenette_training_data_loader,
            val_dataloaders=imagenette_val_data_loader)

