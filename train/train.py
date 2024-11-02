import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(32 * 7 * 7, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 32 * 7 * 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1,output_padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1,output_padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1), 
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 32, 7, 7) 
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def display_latent_samples(model, device, num_samples=20):
    """Display samples from different regions of the latent space"""
    with torch.no_grad():
        # Create figure
        fig = plt.figure(figsize=(15, 15))
        
        # Generate samples from a grid in latent space
        x = np.linspace(-3, 3, num_samples)
        y = np.linspace(-3, 3, num_samples)
        
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                sample = model.decode(z)
                ax = fig.add_subplot(num_samples, num_samples, i * num_samples + j + 1)
                ax.imshow(sample.cpu().numpy()[0, 0], cmap='gray')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('latent_samples.png')
        plt.close()

def visualize_interpolation(model, device, num_steps=10):
    """Display interpolation between two random points in latent space"""
    with torch.no_grad():
        # Generate two random points in latent space
        z1 = torch.randn(1, model.fc_mu.out_features).to(device)
        z2 = torch.randn(1, model.fc_mu.out_features).to(device)
        
        # Create interpolated points
        alphas = np.linspace(0, 1, num_steps)
        fig, axs = plt.subplots(1, num_steps, figsize=(15, 2))
        
        for i, alpha in enumerate(alphas):
            z = alpha * z1 + (1 - alpha) * z2
            sample = model.decode(z)
            axs[i].imshow(sample.cpu().numpy()[0, 0], cmap='gray')
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('interpolation.png')
        plt.close()

def train_vae():
    # Hyperparameters
    latent_dim = 2
    batch_size = 128
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            
            # Reconstruction + KL divergence losses
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            loss = recon_loss + kl_loss * 5
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f'Epoch {epoch}: Loss = {train_loss / len(train_loader.dataset)}')
    
    # Generate latent space visualization data
    model.eval()
    grid_size = 20
    digit_samples = {}
    
    # Create a grid in latent space
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            with torch.no_grad():
                decoded = model.decode(z)
                digit = decoded.cpu().numpy()[0, 0]
                digit_samples[f"{xi:.2f},{yi:.2f}"] = digit.tolist()
    
    # Save model and samples
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim
    }, 'vae_mnist.pth')
    
    with open('latent_samples.json', 'w') as f:
        json.dump(digit_samples, f)

    # After training, generate visualizations
    print("Generating latent space visualizations...")
    display_latent_samples(model, device)
    visualize_interpolation(model, device)

if __name__ == "__main__":
    train_vae()