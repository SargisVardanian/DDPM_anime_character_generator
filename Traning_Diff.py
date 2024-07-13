import torch
from Model import *
from Script import *
from tqdm import tqdm
import argparse
import time
import random
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

print("PyTorch Version:", torch.__version__)
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

def get_subset(dataset, subset_size):
    indices = torch.randperm(len(dataset))[:subset_size]
    return torch.utils.data.Subset(dataset, indices)

def train(model, diffusion, full_dataset, num_epochs, device, optimizer, scheduler=None, channels=3, subset_size=None):
    model.to(device)
    start_time = time.time()
    last_save_time = start_time

    for epoch in range(num_epochs):
        # Get a new subset for each epoch
        if subset_size is not None and subset_size < len(full_dataset):
            train_dataset = get_subset(full_dataset, subset_size)
        else:
            train_dataset = full_dataset

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model.train()
        total_loss = 0
        stepo = 0
        epoch_start_time = start_time
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs} ") as pbar:
            for step, (batch, _) in pbar:  # Ignore labels
                current_time = time.time()

                batch = batch.to(device)
                optimizer.zero_grad()
                stepo += 1
                t = torch.randint(0, diffusion.num_timesteps, (batch.shape[0],), device=device).long()
                loss = diffusion.p_losses(model, batch, t, labels=None)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)  # Calculate average loss

                pbar.set_postfix(loss=loss.item(), avg_loss=avg_loss)  # Add avg_loss to tqdm postfix

        print(f"Epoch {epoch + 1} average loss: {avg_loss}")

        if scheduler is not None:
            scheduler.step()

        # Save model checkpoint every epoch
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, '/Diffusion_96.pt')
            print('Epoch checkpoint saved')
            generate_and_save_images(model, diffusion, num_images=4, image_size=image_size, channels=channels, device=device, save_path=f"generated_images_{epoch}.png")

    print("Training completed.")


dataset_path = '/Users/sargisvardanyan/PycharmProjects/Diffusion_on_transformer/CIFAR'

# Трансформации
transform = transforms.Compose([
    # transforms.RandomRotation(degrees=10),  # Уменьшим вращение до 10 градусов
    # transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # Случайное кадрирование и изменение размера
    transforms.RandomHorizontalFlip(),  # Горизонтальное отражение
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Небольшие изменения цвета
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Нормализация для диапазона [-1, 1]
    transforms.Resize((image_size, image_size)),
    # transforms.ToTensor(),
])

# Создаем датасет и загрузчик данных
dataset = AnimalDataset(dataset_path, transform=transform)
dataset_size = len(dataset)
num_classes = dataset_size
indices = list(range(dataset_size))
random.shuffle(indices)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument('--device', type=str, default='mps', help='Device to use for training (e.g., "cpu", "cuda", "mps")')
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help='number of epochs: ')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else 'cpu')
    num_epochs = args.num_epochs
    print(device)

    # Initialize model, diffusion, dataloader, optimizer, etc.
    model = Diffusion_Model(img_size=image_size, device=device).to(device)
    diffusion = DiffusionModel(num_timesteps=num_timesteps, device=device)

    size_of_model = count_trainable_params(model)
    criterion = nn.MSELoss()
    try:
        checkpoint = torch.load('/Diffusion_96.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print("size_of_model: ", size_of_model)
    except:
        print("No checkpoint found. Starting from scratch.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        print("size_of_model: ", size_of_model)
        start_epoch = 0


    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # add_noise_images(dataloader, diffusion)

    subset_size = int(0.3 * len(dataset))
    # # Train the model
    # train(model, diffusion, dataset, num_epochs=num_epochs, device=device, optimizer=optimizer, channels=num_chan, subset_size=subset_size)

    # # Generate and save images
    generate_and_save_images(model, diffusion, num_images=36, image_size=image_size, channels=num_chan, device=device, save_path="generated_images36.png")
