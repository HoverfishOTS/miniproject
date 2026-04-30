import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    """
    U-Net Generator for Pix2Pix.
    Translates a 3-channel Post-Disaster image into a 3-channel Pre-Disaster image.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # 256x256
        self.down1 = UNetDown(in_channels, 64, normalize=False) # 128x128
        self.down2 = UNetDown(64, 128) # 64x64
        self.down3 = UNetDown(128, 256) # 32x32
        self.down4 = UNetDown(256, 512, dropout=0.5) # 16x16
        self.down5 = UNetDown(512, 512, dropout=0.5) # 8x8
        self.down6 = UNetDown(512, 512, dropout=0.5) # 4x4
        self.down7 = UNetDown(512, 512, dropout=0.5) # 2x2
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5) # 1x1

        self.up1 = UNetUp(512, 512, dropout=0.5) # 2x2
        self.up2 = UNetUp(1024, 512, dropout=0.5) # 4x4
        self.up3 = UNetUp(1024, 512, dropout=0.5) # 8x8
        self.up4 = UNetUp(1024, 512, dropout=0.0) # 16x16
        self.up5 = UNetUp(1024, 256, dropout=0.0) # 32x32
        self.up6 = UNetUp(512, 128, dropout=0.0) # 64x64
        self.up7 = UNetUp(256, 64, dropout=0.0) # 128x128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ) # 256x256

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

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
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    """
    ResNet Generator for Pix2Pix / CycleGAN.
    Translates a 3-channel image to a 3-channel image using 9 Residual Blocks.
    Prevents identity mapping by forcing data through a bottleneck without long skip connections.
    """
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, out_features, 7),
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
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(out_features, out_channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class DiscriminatorPatchGAN(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix.
    Takes paired (Post-Disaster, Pre-Disaster) images concatenated on the channel dimension.
    Outputs a grid of probability values for realism.
    """
    def __init__(self, in_channels=6):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False), # 128x128
            *discriminator_block(64, 128), # 64x64
            *discriminator_block(128, 256), # 32x32
            *discriminator_block(256, 512), # 16x16
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False) # 16x16
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

if __name__ == "__main__":
    # Topology Test
    gen = GeneratorResNet()
    disc = DiscriminatorPatchGAN()
    
    post = torch.randn(2, 3, 256, 256)
    pre = torch.randn(2, 3, 256, 256)
    
    fake_pre = gen(post)
    disc_out = disc(post, fake_pre)
    
    print(f"Post Input Shape: {post.shape}")
    print(f"Generated Pre Shape: {fake_pre.shape}")
    print(f"Discriminator Output Shape: {disc_out.shape}")
