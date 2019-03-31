from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

data_transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
test_dataset = datasets.ImageFolder(root = './test/test_200',transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 32)



class HccrCNN(torch.nn.Module):

	def __init__ (self):
		super(HccrCNN,self).__init__()

		#input channels = 1, output channels = 64
		self.conv1 = torch.nn.Conv2d(3, 64, kernel_size = 3, stride =1, padding =1 )
		self.pool1 = torch.nn.MaxPool2d(kernel_size = 2,stride =2, padding = 0)

		#input chennels = 64, output channels = 128
		self.conv2 = torch.nn.Conv2d(64, 128, kernel_size = 3, stride =1, padding =1 )
		self.pool2 = torch.nn.MaxPool2d(kernel_size = 2,stride =2, padding = 0)

		#input chennels = 128, output channels = 256
		self.conv3 = torch.nn.Conv2d(128, 256, kernel_size = 3, stride =1, padding =1 )
		self.pool3 = torch.nn.MaxPool2d(kernel_size = 2,stride =2, padding = 0)

		#1024 input feature, 200 ouput  classes
		self.fc1 = torch.nn.Linear(256*8*8,1024)
		self.fc2 = torch.nn.Linear(1024,200)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)

		x = F.relu(self.conv2(x))
		x = self.pool2(x)

		x = F.relu(self.conv3(x))
		x = self.pool3(x)

        #flatten (256,8,8)
		x = x.view(-1,256*8*8)

		x = F.relu(self.fc1(x))

		x = self.fc2(x)

		return F.log_softmax(x)

model = HccrCNN()

optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum = 0.5)

checkpoint = torch.load('HccrCNN.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

criterion = torch.nn.CrossEntropyLoss()

model.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    # sum up batch loss
    test_loss += criterion(output, target).data.item()
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
100. * correct / len(test_loader.dataset)))