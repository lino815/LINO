import pytorch_lightning as pl
from .emunet_parts import *
import torch
from torch import optim
from img_lib import plot_net_predictions
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassRecall, MulticlassF1Score, \
    MulticlassPrecision, Dice
from loss.loss import FocalLoss, DiceLoss


class EMUnet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, lr=1e-3, optimizer='SGD', scheduler_name='StepLR', loss_name='CELoss',
                 weight_decay=0.0005):
        super(EMUnet, self).__init__()
        self.n_classes = n_classes
        self.lr = lr
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler_name
        self.loss_name = loss_name
        self.weight_decay = weight_decay
        if self.loss_name == 'CELoss':
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_name == "FocalLoss":
            self.criterion = FocalLoss()
        elif self.loss_name == "DiceLoss":
            self.criterion = DiceLoss(num_classes=n_classes)
        self.val_jaccard = MulticlassJaccardIndex(num_classes=n_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=n_classes)
        self.val_recall = MulticlassRecall(num_classes=n_classes)
        self.val_precision = MulticlassPrecision(num_classes=n_classes)
        self.val_F1 = MulticlassF1Score(num_classes=n_classes)
        self.val_dice = Dice(num_classes=n_classes)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        bilinear = False
        self.up1 = Up_EM(1024, 512, bilinear=bilinear)
        self.up2 = Up_EM(512, 256, bilinear=bilinear)
        self.up3 = Up_EM(256, 128, bilinear=bilinear)
        self.up4 = Up_EM(128, 64, bilinear=bilinear)
        self.outc = OutConv(64 + 128 + 256, n_classes)

    def forward(self, x):
        input = self.inc(x)  # x1
        # print("input.shape: ", input.shape)
        input_half1, input_half2 = torch.chunk(input, 2, dim=1)
        d1 = self.down1(input)  # x2
        # print("d1.shape: ", d1.shape)
        d1_half1, d1_half2 = torch.chunk(d1, 2, dim=1)
        # print(d1.size(), d1_half1.size(), d1_half2.size())
        d2 = self.down2(d1)  # x3
        # print("d2.shape: ", d2.shape)
        d2_half1, d2_half2 = torch.chunk(d2, 2, dim=1)
        d3 = self.down3(d2)  # x4
        # print("d3.shape: ", d3.shape)
        d3_half1, d3_half2 = torch.chunk(d3, 2, dim=1)
        d4 = self.down4(d3)  # x5
        # print("d4.shape: ", d4.shape)
        u1 = self.up1(d4, d3_half1)
        u2 = self.up2(u1, d2_half1)
        u2up = self.upsample4(u2)
        u3 = self.up3(u2, d1_half1)
        u3up = self.upsample2(u3)
        # print("u2up, u2up",u2up.size(), u3up.size())
        u4 = self.up4(u3, input_half1)
        u4 = torch.cat((u2up, u3up, u4), dim=1)
        output = self.outc(u4)
        return torch.sigmoid(output)

    def training_step(self, batch, batch_idx):
        # criterion = nn.CrossEntropyLoss()
        imgs = batch['image']
        # remove for 4 dim images
        if imgs.shape[2] == 4:
            imgs = imgs[:, :, 0:3]
        true_masks = batch['mask']

        imgs = imgs
        true_masks = true_masks

        outputs = self(imgs)
        probs = torch.softmax(outputs, dim=1)
        masks_pred = torch.argmax(probs, dim=1)
        loss = self.criterion(outputs, true_masks)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        if batch_idx % 100 == 0:
            # TODO change here to normal log?
            self.logger.experiment.add_figure('predictions vs. actuals',
                                              plot_net_predictions(imgs, true_masks, masks_pred, imgs.shape[0]),
                                              global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        # criterion = nn.CrossEntropyLoss()
        data = batch['image']
        true_masks = batch['mask']
        preds = self(data)

        loss = self.criterion(preds, true_masks)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

        jaccard = self.val_jaccard(preds, true_masks)
        accuracy = self.val_accuracy(preds, true_masks)
        precision = self.val_recall(preds, true_masks)
        recall = self.val_recall(preds, true_masks)
        f1 = self.val_F1(preds, true_masks)
        dice = self.val_dice(preds, true_masks)

        self.log('val_jaccard', jaccard, on_step=False, on_epoch=True)  # why prog_bar=True?
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.3 * 100), gamma=0.1)
        if self.optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        if self.optimizer_name == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.3 * 100), gamma=0.1)
        if self.scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.3 * 100), gamma=0.1)
        elif self.scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        elif self.scheduler_name == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        else:
            raise ValueError("Invalid scheduler name")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = self.criterion
        self.log("test_loss", test_loss)


