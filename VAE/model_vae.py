import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        
        # Input: 3 x 224 x 224
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # 32 x 112 x 112
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 64 x 56 x 56
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 128 x 28 x 28
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 256 x 14 x 14
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # 512 x 7 x 7
        
        self.flatten_dim = 512 * 7 * 7
        
        # Outputs a mean and a log variance for the latent distribution
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = x.view(x.size(0), -1) # Flatten
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Decoder, self).__init__()
        
        self.flatten_dim = 512 * 7 * 7
        self.fc = nn.Linear(latent_dim, self.flatten_dim)
        
        # Input: 512 x 7 x 7
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # 256 x 14 x 14
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 128 x 28 x 28
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 64 x 56 x 56
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 32 x 112 x 112
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # 3 x 224 x 224
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 7, 7) # Unflatten
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        
        # Use Sigmoid to ensure pixel values are between [0, 1]
        x = torch.sigmoid(self.deconv5(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick: 
        Allows gradients to backpropagate through the random sampling process.
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

if __name__ == "__main__":
    # Test model topology
    model = VAE()
    dummy_input = torch.randn(2, 3, 224, 224) # Batch size 2
    recon, mu, logvar = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent Mean (mu) shape: {mu.shape}")
