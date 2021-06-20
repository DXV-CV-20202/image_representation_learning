import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

cuda = torch.cuda.is_available()

from datasets import IRDataset

mean, std = 0.1307, 0.3081

triplet_train_dataset = IRDataset(descriptor_path = './data/cifar-10/train.json',
                                transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
triplet_test_dataset = IRDataset(descriptor_path = './data/cifar-10/test.json',
                                transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))

batch_size = 1

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

from networks import EmbeddingNetL2, TripletNet
from losses import TripletLoss
from trainer import fit
import torch.nn as nn

embedding_net = EmbeddingNetL2()
triplet_net = TripletNet(embedding_net)

if cuda:
    triplet_net = triplet_net.cuda()

margin = 1.0
loss_fn = TripletLoss(margin)

lr = 1e-3
optimizer = optim.Adam(triplet_net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100

fit(triplet_train_loader, triplet_test_loader, triplet_net, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
torch.save(triplet_net.state_dict(), 'checkpoint.ckp')