# Latent Space Explorer

Online version **[here](https://cpldcpu.github.io/LatentSpaceExplorer/)**.

A web-based interactive tool to explore the latent space of a variational autoencoder (VAE) that was trained on the MNIST dataset. Idea shamelessly taken from [N8s implementation](https://n8python.github.io/mnistLatentSpace/).

This was an exercise in "speed-prompting" starting with a Claude Artifact and then using the "Copilot Edits" interface with Sonnet and o1-preview to complete the artifact and training code. Total time until all functionality was implemented, including training was 2:30h. 

The app is based on Typescript, React, Tailwind, Vite and uses an onnx runtime for inference. The training code uses pytorch. I reused the harness from the [Neural Network Visualizer](https://github.com/cpldcpu/neural-network-visualizer), to speed up deployment. 

The design was all done by claude. I asked for "Cyberpunk" style...

[![LatentSpaceExplorer](screenshot.png)](https://cpldcpu.github.io/LatentSpaceExplorer/)

## How to Use

The app consists of two main parts: The latent space explorer and the VAE model viewer. Use the mouse to pick a point in the latent space, and the VAE model will generate an image from that point. The latent space explorer shows the distribution of the latent space in a 2D projection. The colors indicate the class of the digit

## Neural Network and Training

The training code (Python) can be found in the `train` directory. `train.py` trains the VAE and saves it as a checkpoint in addition to some test-images. `export_vae_2_onnx.py` reads the checkpoint and exports to onnx and saves a latent space make as .msgpack and .json.

The VAE definition is shown below. Interestingly, this was one of the parts that was messed up by Claude, so I had to manually fix the padding and channels. Certainly, a smaller model would have also done a job. Having two layers at full resolution in the decoder turned out to be crucial to avoid too blurry output.

```python
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
```

## Building

The core code can be found in [`webcode/src/pages/index.tsx`](webcode/src/pages/index.tsx). I used [Claude Artifacts Starter](https://github.com/EndlessReform/claude-artifacts-starter) as a harness to deploy the artifact to a github.io page.

All web code is in the `webcode` directory. Read Claude Artifacts Starter's [README](webcode/README.md) for more information.
