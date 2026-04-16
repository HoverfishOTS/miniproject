import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SiameseEncoder(nn.Module):
    """Single branch of the Siamese Network"""
    def __init__(self):
        super().__init__()
        # 3 RGB Channels
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4

class SiameseUNetTransformer(nn.Module):
    def __init__(self, num_classes=4):
        """
        num_classes=4 matches xBD dataset damage labels:
        0: No damage, 1: Minor, 2: Major, 3: Destroyed
        """
        super().__init__()
        # Shared Encoder
        self.encoder = SiameseEncoder()
        
        # Transformer Bottleneck
        # Pre and Post deepest features are concatenated (512 + 512 = 1024 channels)
        self.transformer_dim = 1024
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim, 
            nhead=8, 
            dim_feedforward=2048, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Decoder (Upsampling and analyzing)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # 512 (current) + 256 (pre skip) + 256 (post skip) = 1024
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # 256 + 128 + 128 = 512
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # 128 + 64 + 64 = 256
        self.conv3 = DoubleConv(256, 128)
        
        # Final logits mapping to the 4 classes
        self.outc = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, pre_img, post_img):
        # 1. Siamese Dual Encoding
        pre_x1, pre_x2, pre_x3, pre_x4 = self.encoder(pre_img)
        post_x1, post_x2, post_x3, post_x4 = self.encoder(post_img)
        
        # 2. Combine at bottleneck (B, 1024, H/8, W/8)
        bottleneck = torch.cat([pre_x4, post_x4], dim=1)
        B, C, H_spatial, W_spatial = bottleneck.shape
        
        # 3. Transformer Global Attention
        # Transform spatial map into sequence (B, SequenceLength, Channels)
        seq = bottleneck.view(B, C, -1).permute(0, 2, 1).contiguous()
        tf_out = self.transformer(seq)
        # Transform sequence back to spatial map
        tf_out = tf_out.permute(0, 2, 1).contiguous().reshape(B, C, H_spatial, W_spatial)
        
        # 4. U-Net Decoding with Dual Skip Connections
        x = self.up1(tf_out)
        x = torch.cat([x, pre_x3, post_x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, pre_x2, post_x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, pre_x1, post_x1], dim=1)
        x = self.conv3(x)
        
        # 5. Output segmentation map (B, 4, H, W)
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    # Test topology
    model = SiameseUNetTransformer()
    dummy_pre = torch.randn(2, 3, 256, 256)
    dummy_post = torch.randn(2, 3, 256, 256)
    out = model(dummy_pre, dummy_post)
    print(f"Pre shape: {dummy_pre.shape}")
    print(f"Post shape: {dummy_post.shape}")
    print(f"Output shape (Batch, Classes, H, W): {out.shape}")
