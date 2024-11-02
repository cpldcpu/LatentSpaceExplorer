import torch
import torch.onnx
from torch import nn
import json
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
import msgpack

class VAEDecoder(nn.Module):
    def __init__(self):
        super(VAEDecoder, self).__init__()
        self.decoder_input = nn.Linear(2, 32 * 7 * 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 32, 7, 7)
        x = self.decoder(x)
        return x

def get_latent_representations(vae, device, dataset, batch_size=128):
    """Get latent space representations and corresponding digit classes"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_points = []
    labels = []
    
    vae.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            mu, _ = vae.encode(data)
            latent_points.append(mu.cpu().numpy())
            labels.append(target.numpy())
    
    return np.concatenate(latent_points), np.concatenate(labels)

def create_class_density_map(latent_points, labels, grid_size=50, sigma=0.1):
    """Create density maps for each digit class"""
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    density_maps = np.zeros((10, grid_size, grid_size))
    
    for digit in range(10):
        points = latent_points[labels == digit]
        
        # Create density map for this digit
        for px, py in points:
            gaussian = np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * sigma**2))
            density_maps[digit] += gaussian
            
        # Normalize
        if density_maps[digit].max() > 0:
            density_maps[digit] /= density_maps[digit].max()
    
    return density_maps, x, y

def filter_state_dict(state_dict):
    """Remove unexpected keys from state dictionary"""
    filtered_dict = {}
    expected_prefixes = [
        'encoder.',
        'decoder.',
        'fc_mu.',
        'fc_var.',
        'decoder_input.'
    ]
    
    for key in state_dict:
        # Only keep keys that start with expected prefixes
        if any(key.startswith(prefix) for prefix in expected_prefixes):
            filtered_dict[key] = state_dict[key]
    return filtered_dict
def export_model_and_metadata():
    # Load the trained VAE
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    checkpoint = torch.load('vae_mnist.pth', map_location=device)
    
    # Filter the state dictionary before using it
    filtered_state_dict = filter_state_dict(checkpoint['model_state_dict'])
    
    # Create and load decoder
    decoder = VAEDecoder().to(device)
    
    # Extract decoder weights from filtered state dict
    decoder_state_dict = {
        'decoder_input.weight': filtered_state_dict['decoder_input.weight'],
        'decoder_input.bias': filtered_state_dict['decoder_input.bias']
    }
    
    # Map the decoder layers
    for key in filtered_state_dict:
        if key.startswith('decoder.'):
            decoder_state_dict[key] = filtered_state_dict[key]
    
    decoder.load_state_dict(decoder_state_dict)
    decoder.eval()
    
    # Export decoder to ONNX - ensure dummy input is on the same device
    dummy_input = torch.randn(1, 2).to(device)
    torch.onnx.export(
        decoder,
        dummy_input,
        "decoder.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['latent_vector'],
        output_names=['generated_image'],
        dynamic_axes={
            'latent_vector': {0: 'batch_size'},
            'generated_image': {0: 'batch_size'}
        }
    )
    
    # Get latent space class information
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    
    # Load full VAE for encoding
    from train import VAE  # Import your VAE class
    vae = VAE(latent_dim=2).to(device)
    vae.load_state_dict(filtered_state_dict)
    vae.eval()
    
    # Get latent representations and create density maps
    latent_points, labels = get_latent_representations(vae, device, dataset)
    density_maps, x, y = create_class_density_map(latent_points, labels)

    # Save metadata
    metadata = {
        'x_coords': x.tolist(),
        'y_coords': y.tolist(),
        'density_maps': density_maps.tolist(),
        'latent_points': latent_points.tolist(),
        'labels': labels.tolist()
    }
    
    # with open('vae_metadata.json', 'w') as f:
    #     json.dump(metadata, f)

    # MessagePack export
    with open('vae_metadata.msgpack', 'wb') as f:
        msgpack.pack(metadata, f)

if __name__ == "__main__":
    export_model_and_metadata()