import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# dataset
train_data = torchvision.datasets.CIFAR10("dataset", train=True, transform=torchvision.transforms.ToTensor(), download = False)
test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(), download = False) 

# len
train_data_size = len(train_data)
test_data_size = len(test_data)

print("len of train: {}".format(train_data_size))
print("len of test: {}".format(test_data_size))

# use dataloader to load dataset
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

model = Model()

# loss func
loss_fn = nn.CrossEntropyLoss()

# optim
optim = torch.optim.SGD(model.parameters(), lr = 0.01)

# train
total_train_step = 0
total_test_step = 0
epoch = 10

# summary
writer = SummaryWriter("logs")

for i in range(epoch):
    print("------------------{}th training epoch-----------------".format(i+1))

    # model.train()  ## for some layers like batchnorm or dropout
    for data in train_dataloader:
        imgs, targets = data
        outputs = model(imgs)

        loss = loss_fn(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if(total_train_step % 100 == 0):
            print("{}th training step: loss = {}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    # test 
    # model.eval() ## same as line 60
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()     ## 横向来看，最大权重的class等于正确class的个数之和

    print("loss on test dataset: {}".format(total_test_loss))
    print("accuracy on test dataset: {}".format(total_accuracy))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(model.state_dict(), "model/model_{}.pth".formart(i))
    print("model saved") 

writer.close()