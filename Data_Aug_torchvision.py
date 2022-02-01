import torch
import torch.vision.transformers as trasforms
from torchvision.utils import save_image
from customDataset import CatsAndDogsDataset

#Load Data
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.5),   #rand change in brightness
    transforms.RandomRotation(degrees=450),   
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    #improves trainnig 
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])     #(value-mean)/std
])

dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir = 'cats_dogs_resized',
                            transform = my_transforms)

#train_loader = data_loader

img_num = 0
for _ in range(10):
    for img, label in dataset:
        #print(img.shape)
        save_image(img, 'img'+str(img_num)+ '.png')
        img_num += 1





