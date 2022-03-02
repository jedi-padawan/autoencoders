#fsize = 64
fsize = 192
bsize = 64
#direc = "./issm0/"
#direc = "./SEM_haikei/"
data_path = '/pool/data/ISSM2020/issm2020-ai-challenge-normal-only/'


from torchvision import transforms
from torchvision.transforms import RandAugment

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(fsize),
        transforms.CenterCrop(fsize),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #transforms.Normalize([0.35,], [0.5,])
    ]),
    'test': transforms.Compose([
        transforms.Resize(fsize),
        transforms.CenterCrop(fsize),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #transforms.Normalize([0.321, 0.321, 0.321], [0.262, 0.262, 0.262])
    ]),
}
data_transforms["train"].transforms.insert(0, RandAugment(12, 5, 31))


from torchvision import datasets
from torch.utils.data import DataLoader


# すべて
# image_datasets = datasets.ImageFolder(data_path + "semTrain/" + "semTrain/", data_transforms["train"])
# image_datasets2 = datasets.ImageFolder(data_path + "semTest/" + "semTest/", data_transforms["test"])

# 正常のみ
image_datasets = datasets.ImageFolder(data_path + "semTrain/" + "semTrain/", data_transforms["train"])
image_datasets2 = datasets.ImageFolder(data_path + "semTest/" + "semTest/", data_transforms["test"])

train_loader = DataLoader(image_datasets,
                          batch_size=bsize, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
test_loader = DataLoader(image_datasets2,
                         batch_size=1, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)

import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, n_channel_base, d_code_space):
        super(encoder, self).__init__()
        self.n_channel_base = n_channel_base
        self.d_code_space = d_code_space
        main = nn.Sequential(
            nn.Conv2d(3, n_channel_base, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_channel_base, 2 * n_channel_base, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * n_channel_base, 4 * n_channel_base, 3, 2, padding=1),
            nn.LeakyReLU(),
            )
        self.main = main
        self.linear = nn.Linear(4 * 4 * 4 * 4 * n_channel_base, self.d_code_space)

    def forward(self, input):
        #print(input[1].shape)
        output = self.main(input)
        #print(output[1].shape)
        output = output.view(-1, 4*4*4*4*self.n_channel_base)
        #print(output[1].shape)
        output = self.linear(output)
        #print(output[1].shape)
        return output
    
class decoder(nn.Module):
    def __init__(self, n_channel_base, d_code_space):
        super(decoder, self).__init__()
        self.n_channel_base = n_channel_base
        self.d_code_space = d_code_space
        preprocess = nn.Sequential(
            nn.Linear(self.d_code_space, 4 * 4 * 4 * self.n_channel_base),
            nn.BatchNorm1d(4 * 4 * 4 * self.n_channel_base),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.n_channel_base, 2 * self.n_channel_base, 2, stride=2),
            nn.BatchNorm2d(2 * self.n_channel_base),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.n_channel_base, self.n_channel_base, 2, stride=2),
            nn.BatchNorm2d(self.n_channel_base),
            nn.ReLU(True),
        )
        
        deconv_out = nn.ConvTranspose2d(self.n_channel_base, 4 * 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        #print(input[1].shape)
        output = self.preprocess(input)
        #print(output[1].shape)
        output = output.view(-1, 4 * self.n_channel_base, 4, 4)
        #print(output[1].shape)
        output = self.block1(output)
        #print(output[1].shape)
        output = self.block2(output)
        #print(output[1].shape)
        output = self.deconv_out(output)
        #print(output[1].shape)
        output = self.tanh(output)
        #print(output[1].shape)
        return output.view(-1, 3, fsize, fsize )


### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


import torch
import torch.optim as optim


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

enc = encoder(n_channel_base=2*fsize, d_code_space=64).to(device)
dec = decoder(n_channel_base=2*fsize, d_code_space=64).to(device)
optimizer1 = optim.Adam(enc.parameters(), lr = 0.001)
optimizer2 = optim.Adam(dec.parameters(), lr = 0.001)

loss_fn = nn.MSELoss()

losses = []         #loss_functionの遷移を記録
n_epochs  = 500

import pickle

for epoch in range(n_epochs):
    running_loss = 0.0  
    enc.train()         #ネットワークをtrainingモード
    dec.train() 

    for i, (image,_) in enumerate(train_loader):
        image = image.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        XX_pred = enc(image)                  #ネットワークで予測
        XX_pred = dec(XX_pred)
        #XX_pred = XX_pred.to(device)
        loss = loss_fn(image, XX_pred)   #予測データと元のデータの予測
        loss.backward()
        optimizer1.step()              #勾配の更新
        optimizer2.step()
        running_loss += loss.item()

    losses.append(running_loss / (i + 1))
    print("epoch", epoch+1, ": ", running_loss / (i + 1))
    model_path = 'cae_encoder.pth'
    torch.save(enc.state_dict(), model_path)
    model_path = 'cae_decoder.pth'
    torch.save(dec.state_dict(), model_path)
    with open('losses.pickle', 'wb') as p:
        pickle.dump(losses, p)


import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
