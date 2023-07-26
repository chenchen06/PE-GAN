from torchvision import datasets, transforms
from gan_models import Generator, Discriminator
from gan_train import train

def main():
    # Create a device object for the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR10 dataset
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Create models
    G = Generator()
    D = Discriminator()

    # Train the model
    train(D, G, trainloader, device=device)

    # Generate some images and show them
    G = G.eval()
    with torch.no_grad():
        # Get a batch of real images
        real_images, _ = next(iter(trainloader))
        real_images = real_images.to(device)
        
        # Generate fake images
        fake_images = G(real_images)

        # Move the images back to CPU and convert to numpy arrays
        real_images = real_images.cpu().numpy()
        fake_images = fake_images.cpu().numpy()

        # Show images
        for i in range(4):
            plt.subplot(2, 4, i + 1)
            plt.imshow(real_images[i])
            plt.subplot(2, 4, i + 5)
            plt.imshow(fake_images[i])
        plt.show()

if __name__ == '__main__':
    main()
