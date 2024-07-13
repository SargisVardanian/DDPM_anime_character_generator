from Model import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Гиперпараметры
batch_size = 4
learning_rate = 2e-5
num_chan = 3
num_epochs = 520
image_size = 96
num_timesteps = 1000
patch_size = 4
embed_dim = 128
num_heads = 8
depth = 6


class DiffusionModel:
    def __init__(self, num_timesteps=1000, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = self.linear_beta_schedule().to(self.device)
        self.alphas = (1. - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(self.device)

    def linear_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.005
        return torch.linspace(beta_start, beta_end, self.num_timesteps).to(self.device)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t).to(self.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).to(self.device)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, labels, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        _, predicted_noise = denoise_model(x_noisy, t, labels)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, labels=None):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        device = torch.device('mps')
        model = model.to(device)
        x = x.to(device)
        t = t.to(device)
        # labels = labels.to(device)
        _, x_m = model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * x_m / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x).to(self.device)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels=None):
        device = next(model.parameters()).device

        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, label=None):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def add_noise_images(dataloader, diffusion_model, num_timesteps=1000):
    device = diffusion_model.device
    for images, labels in dataloader:
        imageo = images[0].to(device)  # Move the image to the correct device
        break

    for t in range(1, num_timesteps, 100):
        t_tensor = torch.tensor([t], device=device)
        print('imageo: ', imageo.shape)
        noise = torch.randn_like(imageo.unsqueeze(0), device=device)
        image = diffusion_model.q_sample(imageo.unsqueeze(0), t_tensor, noise)
        print('image: ', image.shape, 'noise: ', noise.shape)

        # Convert images for visualization
        img = image[0].cpu().permute(1, 2, 0).numpy()  # Transpose to (H, W, C)
        old_img = noise[0].cpu().permute(1, 2, 0).numpy()  # Transpose to (H, W, C)

        # Display the images and noise
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow((img * 127.5 + 127.5).astype(np.uint8))
        axs[0].set_title(f'Noisy Image at t={t}')
        axs[1].imshow((old_img * 127.5 + 127.5).astype(np.uint8))
        axs[1].set_title(f'Noise at t={t}')
        plt.show()


def generate_and_save_images(model, diffusion, num_images, image_size, channels, device, save_path):
    model.eval()
    samples = diffusion.sample(model, image_size=image_size, batch_size=num_images, channels=channels)

    # Create a grid of images
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # Ensure axes is always a 2D array
    if grid_size == 1:
        axes = np.array([[axes]])
    elif grid_size > 1 and axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for i, img in enumerate(samples[-1]):
        if i >= num_images:
            break
        if isinstance(img, np.ndarray):
            img = torch.tensor(img)  # Ensure img is a tensor if it is not
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        ax = axes[i // grid_size, i % grid_size]
        ax.imshow(img)
        ax.axis('off')

    for i in range(num_images, grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
