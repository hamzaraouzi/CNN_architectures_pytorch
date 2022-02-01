
import config
import  torch
from tqdm import tqdm
from dataset import get_loaders
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import plot_to_tensorboard, save_checkpoint, load_checkpoint
import torch.nn as nn

from GoogleNet import GoogleNet

def main():
    
    print(f"creation of {config.MODEL} Architecture")

    model = GoogleNet(in_channels=config.IN_CHANNELS,aux_logits=True, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #getting train and test loaders
    train_loader, test_loader = get_loaders()



    #loading model in case we want to continue training
    if config.LOAD_MODEL:
        load_checkpoint(model=model, optimizer=optimizer, lr=config.LEARNING_RATE)

    writer = SummaryWriter(config.LOGS_PATH)
    tensorboard_step = 0
    old_accuracy = 0

    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: ", epoch)
        model.aux_logits = True
        train_loss = train_fn(model, loader=train_loader, optimizer=optimizer)
        model.aux_logits = False
        test_loss, accuracy = test(model=model, test_loader=test_loader)

        plot_to_tensorboard(writer=writer, train_loss=train_loss, test_loss=test_loss, test_accuracy=accuracy,
                            tensorboard_step=tensorboard_step)

        tensorboard_step += 1

        print("accuracy = ", accuracy, " old_accurcy =", old_accuracy)
        if accuracy > old_accuracy:
            file_name = config.MODEL +".pth.tar"
            save_checkpoint(model=model, optimizer=optimizer, file_name=file_name)
            old_accuracy = accuracy

def train_fn(model, loader, optimizer):

    loop = tqdm(loader, leave=True)
    model.train()
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.to(device=config.DEVICE)

        optimizer.zero_grad()
        predictions, aux = model(data)
        
        loss1 = nn.CrossEntropyLoss()(predictions, targets)
        loss2 = nn.CrossEntropyLoss()(aux, targets)
        loss = loss1 + 0.3*loss2
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())

    return train_loss/len(loader.dataset)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        loop = tqdm(test_loader, leave=True)
        for _, (data, target) in enumerate(loop):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            output = model(data)

            loss = nn.CrossEntropyLoss()(output, target)
            test_loss += loss.item()

            # output shape: [batch_size, num_classes]
            _, predicted = output.max(1)
            # predicted shape: [batch_size, 1]

            correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        loop.set_postfix(test_loss=test_loss, accuracy=accuracy)
        return test_loss, accuracy