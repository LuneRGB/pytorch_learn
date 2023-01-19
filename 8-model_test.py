from PIL import Image
import torchvision
from torch import nn
import torch

image_path = "test.png"
img = Image.open(image_path)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
img = transform(img)
print(img.shape)

# nn
class Model(nn.Module):
    def __init__(self): 
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("model/model_0.pth", map_location=torch.device('cpu'))
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
output = model(img)
print(output)

print(output.argmax(1))