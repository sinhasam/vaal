from torchvision import datasets, transforms

def imagenet_transformer():
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def cifar_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
       ])
