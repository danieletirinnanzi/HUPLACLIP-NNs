from torchvision import transforms

# Define the transform for VGG16
VGG16transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels
        transforms.ToTensor(),  # Convert to PyTorch tensor
    ]
)
