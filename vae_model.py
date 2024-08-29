import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any
import matplotlib.pyplot as plt

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

class VanillaVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, hidden_dims=None, input_height=28, input_width=28, log_var_bound=10.0):
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.log_var_bound = log_var_bound  # Bound for log_var
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Calculate the output dimensions of the encoder
        self._calculate_encoder_output_dims()

        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.encoder_output_dim, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_dim)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels, kernel_size=3, padding=1),
            # nn.Tanh(),
            )

    def _calculate_encoder_output_dims(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)
            sample_output = self.encoder(sample_input)
            self.encoder_output_dim = sample_output.numel()
            self.encoder_output_shape = sample_output.shape[1:]  # Save the shape for later use

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        log_var = self.log_var_bound * torch.tanh(log_var)  # Bound log_var
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, *self.encoder_output_shape)  # Use the saved shape
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        print(f"recon_loss: {recon_loss}, KLD: {KLD}")
        return {'loss': recon_loss + KLD}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, num_steps: int = 10) -> List[torch.Tensor]:
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        interpolations = []
        for alpha in torch.linspace(0, 1, num_steps):
            z = mu1 * (1 - alpha) + mu2 * alpha
            interpolations.append(self.decode(z).cpu().detach())
        return interpolations

    def visualize_interpolation(self, x1: torch.Tensor, x2: torch.Tensor, num_steps: int = 10):
        interpolations = self.interpolate(x1, x2, num_steps)
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 2))
        for i, img in enumerate(interpolations):
            axes[i].imshow(img.squeeze(), cmap='gray')
            axes[i].axis('off')
        plt.show()