import torch.nn as nn
import torch.nn.functional as F
import torch
from Model.unetPlusPlus import unetPlusPlus
from torch.nn.modules.padding import ReplicationPad2d
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        # Initial convolution block

        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################

class discriminator_block(nn.Module):

    def __init__(self, in_filters, out_filters, normalize=True):
        super(discriminator_block, self).__init__()

        layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
        if normalize:
            self.layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)


       # self._init_weight()

    def forward(self, x):
        return self.layers(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
        self.CD_model = unetPlusPlus(6,2)

        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.D = nn.Conv2d(512, 1, 4, padding=1)


    def forward(self, img):
        D_logit1, output5 = self.CD_model(img)
        D_logit = self.D(self.pad(D_logit1))
        return [D_logit,output5]

# class Discriminator(nn.Module):
#     def __init__(self, input_shape):
#         super(Discriminator, self).__init__()
#
#         channels, height, width = input_shape
#
#         # Calculate output shape of image discriminator (PatchGAN)
#         self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
#
#         self.conv11 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
#         self.bn11 = nn.BatchNorm2d(32)
#         self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.bn12 = nn.BatchNorm2d(32)
#
#         self.db1 = discriminator_block(32, 64, normalize=False)
#         self.db2 = discriminator_block(64, 128, normalize=False)
#         self.db3 = discriminator_block(128, 256, normalize=False)
#
#         self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=2, output_padding=1)
#         self.conv43d = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
#         self.bn43d = nn.BatchNorm2d(256)
#         self.conv42d = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
#         self.bn42d = nn.BatchNorm2d(256)
#         self.conv41d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
#         self.bn41d = nn.BatchNorm2d(128)
#
#         self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
#         self.conv33d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
#         self.bn33d = nn.BatchNorm2d(128)
#         self.conv32d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
#         self.bn32d = nn.BatchNorm2d(128)
#         self.conv31d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
#         self.bn31d = nn.BatchNorm2d(64)
#
#
#         self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
#         self.conv23d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
#         self.bn23d = nn.BatchNorm2d(64)
#         self.conv22d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
#         self.bn22d = nn.BatchNorm2d(64)
#         self.conv21d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
#         self.bn21d = nn.BatchNorm2d(32)
#
#         self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)
#         self.conv12d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
#         self.bn12d = nn.BatchNorm2d(32)
#         self.conv11d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
#         self.bn11d = nn.BatchNorm2d(16)
#
#         self.conv10d = nn.ConvTranspose2d(16, 2, kernel_size=3, padding=1)
#
#
#         self.sm = nn.LogSoftmax(dim=1)
#
#
#         self.db4 = discriminator_block(256, 512, normalize=False)
#         self.pad = nn.ZeroPad2d((1, 0, 1, 0))
#         self.class_d = nn.Conv2d(512, 1, 4, padding=1)
#
#
#
#
#
#
#     def forward(self, img):
#
#         x11 = F.relu(self.bn11(self.conv11(img)))
#         x12 = F.relu(self.bn12(self.conv12(x11)))
#
#         db_block1 = self.db1(x12)
#         db_block2 = self.db2(db_block1)
#         db_block3 = self.db3(db_block2)
#
#         x4p = F.max_pool2d(db_block3, kernel_size=2, stride=2)
#
#
#         x4d = self.upconv4(x4p)
#
#         pad4 = ReplicationPad2d((0, db_block3.size(3) - x4d.size(3), 0, db_block3.size(2) - x4d.size(2)))
#         x4d = torch.cat((pad4(x4d), db_block3), 1)
#         x43d = F.relu(self.bn43d(self.conv43d(x4d)))
#         x42d = F.relu(self.bn42d(self.conv42d(x43d)))
#         x41d = F.relu(self.bn41d(self.conv41d(x42d)))
#
#         x3d = self.upconv3(x41d)
#         pad3 = ReplicationPad2d((0, db_block2.size(3) - x3d.size(3), 0, db_block2.size(2) - x3d.size(2)))
#         x3d = torch.cat((pad3(x3d), db_block2), 1)
#         x33d = F.relu(self.bn33d(self.conv33d(x3d)))
#         x32d = F.relu(self.bn32d(self.conv32d(x33d)))
#         x31d = F.relu(self.bn31d(self.conv31d(x32d)))
#
#
#         x2d = self.upconv2(x31d)
#         pad2 = ReplicationPad2d((0, db_block1.size(3) - x2d.size(3), 0, db_block1.size(2) - x2d.size(2)))
#         x2d = torch.cat((pad2(x2d), db_block1), 1)
#         x23d = F.relu(self.bn23d(self.conv23d(x2d)))
#         x22d = F.relu(self.bn22d(self.conv22d(x23d)))
#         x21d = F.relu(self.bn21d(self.conv21d(x22d)))
#
#         x1d = self.upconv1(x21d)
#         pad1 = ReplicationPad2d((0, x12.size(3) - x1d.size(3), 0, x12.size(2) - x1d.size(2)))
#         x1d = torch.cat((pad1(x1d), x12), 1)
#         x12d = F.relu(self.bn12d(self.conv12d(x1d)))
#         x11d = F.relu(self.bn11d(self.conv11d(x12d)))
#
#         x10 = self.conv10d(x11d)
#         cd_pred = self.sm(x10)
#
#         db_block4 = self.db4(db_block3)
#         pad0 = self.pad(db_block4)
#         D_logit = self.class_d(pad0)
#
#         return [D_logit,cd_pred]
