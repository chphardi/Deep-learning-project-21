import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image, ImageColor
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random
from torchmetrics.classification import Dice
import torch.optim as optim

class UNet_test(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        # input 256*256*3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # 256*256*3
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # 256*256*64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 128*128*64
        
        # input 128*128*64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 128*128*128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # 128*128*128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64*64*128
        
        # input 64*64*128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # 64*64*256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # 64*64*256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 32*32*256
        
        # input 32*32*256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # 32*32*512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # 32*32*512
        
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
        
    def forward(self ,x):
        # encoder
        xe11 = nn.ReLU(self.e11(x))
        xe12 = nn.ReLU(self.e12(xe11))
        xp1 = self.pool1(xe12)
        
        xe21 = nn.ReLU(self.e21(xp1))
        xe22 = nn.ReLU(self.e22(xe21))
        xp2 = self.pool2(xe22)
        
        xe31 = nn.ReLU(self.e31(xp2))
        xe32 = nn.ReLU(self.e32(xe31))
        xp3 = self.pool3(xe32)
        
        xe41 = nn.ReLU(self.e41(xp3))
        xe42 = nn.ReLU(self.e42(xe41))
        
        # decoeder
        xu1 = self.upconv1(xe42)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = nn.ReLU(self.d11(xu11))
        xd12 = nn.ReLU(self.d12(xd11))
        
        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = nn.ReLU(self.d21(xu22))
        xd22 = nn.ReLU(self.d22(xd21))
        
        xu3 = self.upconv3(xd32)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = nn.ReLU(self.d31(xu33))
        xd32 = nn.ReLU(self.d32(xd31))
        
        out = self.outconv(xd32)
        
        return out
    
class CarDataset(Dataset):
    def __init__(self, root, file_list: list=None, backgrounds: list=None):

        self.root = root
        self.filenames = os.listdir(self.root) if file_list is None else file_list
        if backgrounds is not None:
          self.backgrounds = backgrounds
        else:
          self.backgrounds = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        arr = np.load(os.path.join(self.root, filename))
        car = arr[:,:,0:3]
        labels = arr[:,:,3]

        if self.backgrounds is not None:
          background = random.choices(self.backgrounds)
          car = add_background(car, labels, background)

        car = car.astype(np.float32)
        # car = np.moveaxis(car, 2, 0)

        return car, labels/10

def display_car(data_arr):
    # Can take both full data and already split data
    if data_arr.shape[2] > 3: data_arr = data_arr[:,:,:3]
    img = Image.fromarray(data_arr)
    img.show() # img.show() for jupyter
    
def display_labels(data_arr, parts):
    # Can take both full data and already split data
    if data_arr.ndim > 2: data_arr = data_arr[:,:,3]
    img = Image.fromarray(data_arr)
    pixels = list(img.getdata())
    pixels = [ImageColor.getrgb(parts.get(pixel)['col']) for pixel in pixels]
    image = Image.new("RGB", (256, 256), (0,0,0))
    image.putdata(pixels)
    image.show()
    
def add_background(car, labels, background):
    mask = (labels == 0)
    updated_car = np.where(mask[..., None], background, car)
    updated_car = updated_car.squeeze()

    return updated_car

def load_images_from_folder(folder_path, resize_shape=(256, 256), limit=100):
    image_list = []
    count = 0

    for filename in os.listdir(folder_path):
        # Check if the file is an image file (you can add more extensions if needed)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            file_path = os.path.join(folder_path, filename)

            # Open the image using PIL
            img = Image.open(file_path)

            # Resize the image
            img = img.resize(resize_shape)

            # Convert the image to a numpy array and append to the list
            image_array = np.array(img)
            image_list.append(image_array)

            count += 1
            if count >= limit:
                break

    return image_list
    
if __name__ == "__main__":
    ds = CarDataset("./carseg_data/carseg_data/arrays")
    car, label = ds[1000]
    
    parts = {10: {'col': 'orange', 'name':'hood'}, 
         20: {'col':'darkgreen', 'name':'front door'}, 
         30: {'col':'yellow', 'name':'rear door'}, 
         40: {'col':'cyan', 'name':'frame'}, 
         50: {'col':'purple', 'name':'rear quarter panel'}, 
         60: {'col':'lightgreen', 'name':'trunk lid'}, 
         70: {'col':'blue', 'name':'fender'}, 
         80: {'col':'pink', 'name':'bumper'}, 
         90: {'col':'darkgray', 'name':'rest of car'},
         0 : {'col':'black', 'name':'background'}}

    print(car.shape)
    
    folder_path = './carseg_data/carseg_data/images/landscapes'
    backgrounds = load_images_from_folder(folder_path, limit=250)
    
    black_car = []
    orange_car = []
    photos = []
    for file in os.listdir('./carseg_data/carseg_data/arrays'):
        if 'orange' in file: orange_car.append(file)
        elif 'black' in file: black_car.append(file)
        elif 'photo' in file: photos.append(file)

    print(len(black_car), len(orange_car), len(photos))
    
    root = './carseg_data/carseg_data/arrays'

    photo_train, photo_test = train_test_split(photos, test_size=0.4, random_state=42)
    black_train, black_test = train_test_split(black_car, test_size=0.4, random_state=42)
    orange_train, orange_test = train_test_split(orange_car, test_size=0.4, random_state=42)

    photo_val, photo_test = train_test_split(photo_test, test_size=0.5, random_state=42)
    black_val, black_test = train_test_split(black_test, test_size=0.5, random_state=42)
    orange_val, orange_test = train_test_split(orange_test, test_size=0.5, random_state=42)
    
    joint_train_ds = CarDataset(root, photo_train*3+black_train+orange_train)
    joint_train_background_ds = CarDataset(root, black_train+orange_train, backgrounds)
    joint_val_ds = CarDataset(root, photo_val+black_val+orange_val)

    joint_test_ds = CarDataset(root, photo_test+black_test+orange_test)
    photo_test_ds = CarDataset(root, photo_test)
    black_test_ds = CarDataset(root, black_test)
    orange_test_ds = CarDataset(root, orange_test)


    train_loader = DataLoader(joint_train_ds+joint_train_background_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(joint_val_ds, batch_size=16)
    test_loader = DataLoader(joint_test_ds, batch_size=16)

    photo_test_loader =  DataLoader(photo_test_ds, batch_size=16)
    black_test_loader =  DataLoader(black_test_ds, batch_size=16)
    orange_test_loader =  DataLoader(orange_test_ds, batch_size=16)
    
    dice = Dice(average='micro')
    
    device = "cpu"
    model = UNet_test(10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
    num_epochs = 75
    dice = Dice(average='micro')
    loaders = [train_loader, ]

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        dice_scores_train = []


        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.float()
            labels = labels.long().to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            total_train_loss += loss.item()

            # Calculate dice
            _, pred = torch.max(outputs, 1)
            pred_cpu = pred.to('cpu')
            labels_cpu = labels.to('cpu')

            dice_scores_train.append(dice(pred_cpu, labels_cpu))


            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        dice_scores_val = []

        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.float()
            labels = labels.long().to(device)

            with torch.no_grad():
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

                # Calculate dice
                _, pred = torch.max(outputs, 1)
                pred_cpu = pred.to('cpu')
                labels_cpu = labels.to('cpu')

                dice_scores_val.append(dice(pred_cpu, labels_cpu))


        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {(total_train_loss / len(train_loader)):.4f}, Train dice: {np.mean(dice_scores_train):.4f}, Val Loss: {(total_val_loss / len(val_loader)):.4f}, Val dice: {np.mean(dice_scores_val):.4f}")
