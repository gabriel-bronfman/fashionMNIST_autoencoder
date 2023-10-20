import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
import os
import plotly.express as px
import pandas as pd


from AutoEncoder_CNN import AE_CNN

# Setting to force retraining
TRAIN = False


feature_dim = 8*2*2

#Dictionary with labels for data set:

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

#import, cache, and load Fashion MNIST Dataset
tensor_transform = transforms.ToTensor()
dataset = datasets.FashionMNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = tensor_transform)
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = 16,
                                     shuffle = True)
                                     
#Currently training on a Macbook Pro using metal cores, can accomodate Cuda cores
try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception as e:
    print(e)
    if torch.cuda.is_available():
        device = "cuda:0" 

if not device:
    device = "cpu"
# Model and Settings


#If model exists, load model

if os.path.exists("AutoEncoder.pt") and not TRAIN:
    model = torch.load("AutoEncoder.pt").to(device)
else:
    #Train Model for 20 epochs
    
    model = AE_CNN().to(device)
    model.train()
    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = 9e-2,
                                 weight_decay = 1e-8)

    

    epochs = 20
    
    outputs = []
    losses = []
    
    for epoch in tqdm(range(epochs)):
        for (image,_) in loader:
            total_losses = 0


            image = image.to(device)
            

            reconstructed = model(image)
            
            loss = loss_function(reconstructed, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)

            total_losses += loss.item()


        print("Average Loss: ", total_losses/len(loader))
        print("Done with Epoch ", epoch)

        outputs.append((epoch,image,reconstructed))

    torch.save(model, "AutoEncoder.pt")   
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses[-100:])

    plt.show()


test_dataset = datasets.FashionMNIST(root = "./data",
                         train = False,
                         download = True,
                         transform = tensor_transform)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                     batch_size = 32,
                                     shuffle = True)

features = []
tags = []
model.eval()

# Pass through validation set, capturing list of flattened latent feature space, and creating a list of index-matched ground truth labels
for (image,labels) in test_loader:
    
    image = image.to(device)
    
    batch_features = model.encoder(image).tolist()
    
    for row in batch_features:
            row = np.reshape(row,feature_dim)
            features.append(row)
            
    for label in labels.numpy():
        tags.append(labels_map[label])
        

print(features)
# Using t-stochastic nearest neighbor embedding, we can lower the 1x(8*4*4) space into a 2D space for visualization

tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(np.array(features))

# Create a Pandas dataframe to make graphing easier with Plotly
df = pd.DataFrame(projections, columns=['x', 'y'])
df['labels'] = tags

fig = px.scatter(
    df, x='x', y='y', color='labels'
)

fig.show()

