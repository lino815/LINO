import torch
import torch.nn as nn
from einops import rearrange
from transunet.vit import ViT
from transunet.common import CBR, CB
import pytorch_lightning as pl
from torch import optim
from img_lib import plot_net_predictions
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassRecall, MulticlassF1Score, \
    MulticlassPrecision, Dice
from loss.loss import FocalLoss, DiceLoss
from transunet.convnext import convnext_base

# from vit import ViT
# from common import CBR, CB


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        self.downsample = CB(in_channels, out_channels, k=1, s=stride)
        width = int(out_channels * (base_width / 64))
        self.cbr1 = CBR(in_channels, width, k=1, s=1)
        self.cbr2 = CBR(width, width, k=3, s=2)
        self.cb = CB(width, out_channels, k=1, s=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.cbr2(self.cbr1(x))
        x = self.cb(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            CBR(in_channels, out_channels, k=3, s=1),
            CBR(out_channels, out_channels, k=3, s=1),
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.cbr1 = CBR(in_channels, out_channels, k=7, s=2, p=3)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)
        self.cbr2 = CBR(out_channels * 8, 512, k=3, s=1, p=1)
        self.convnext = convnext_base(pretrained=True)
    def forward(self, x):
        # x1 = self.cbr1(x)
        # print("x1.shape", x1.shape)
        # x2 = self.encoder1(x1)
        # print("x2.shape", x2.shape)
        # x3 = self.encoder2(x2)
        # print("x3.shape", x3.shape)
        # x = self.encoder3(x3)
        # print("x.shape", x.shape)
        x1, x2, x3, x4 = self.convnext(x)
        x = self.vit(x4)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.cbr2(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x


class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels, class_num)


    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        return x


class TransUnetLightning(pl.LightningModule):
    def __init__(self, n_channels, n_classes, lr=1e-3, optimizer='SGD', scheduler_name='StepLR', loss_name='CELoss',
                 weight_decay=0.0005):
        super(TransUnetLightning, self).__init__()
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
        self.dice = Dice(num_classes=n_classes)
        # image shaoe
        self.transunet = TransUNet(256,
                                   n_channels,
                                   out_channels=128,
                                   head_num=4,
                                   mlp_dim=512,
                                   block_num=8,
                                   patch_dim=16,
                                   class_num=n_classes)

    def forward(self, x):
        x = self.transunet(x)
        return torch.sigmoid(x)

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
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        precision = self.val_recall(preds, true_masks)
        recall = self.val_recall(preds, true_masks)
        dice = self.dice(preds, true_masks)

        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        self.log('val_dice', dice, on_epoch=True)

    def configure_optimizers(self, lr=1e-3):
        if self.optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        if self.optimizer_name == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0005)
        if self.optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
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
        self.decoder(z)
        test_loss = self.criterion

        self.log("test_loss", test_loss)


# if __name__ == '__main__':
#     import torch
#
#     transunet = TransUNet(img_dim=512,
#                           in_channels=3,
#                           out_channels=128,
#                           head_num=4,
#                           mlp_dim=512,
#                           block_num=8,
#                           patch_dim=16,
#                           class_num=2)
#
#     print(sum(p.numel() for p in transunet.parameters()))
#     print(transunet(torch.randn(1, 3, 512, 512)).shape)
