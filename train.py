#train.py for CC-Net
#CC-Net is a classification model for handwriting(PD and HC) recognition

import os,json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms, datasets, utils
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from tensorboardX import SummaryWriter
from PIL import Image

from cc_net_model import MyNet2

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    tbwriter=SummaryWriter(log_dir="./logs")
    
    #processing before training
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(227),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.5],[0.5])]),
        "val": transforms.Compose([transforms.Resize((227, 227)),  # cannot 224, must (227, 227)
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5],[0.5])])}
    #input path:train and validate
    data_path = os.path.abspath(os.path.join(os.getcwd(), "./data/"))
    assert os.path.exists(data_path), "{} path does not exist.".format(image_path)
    
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    
    validate_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)

    #'HC':0, 'PD':1
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    #json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices_change.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    if os.path.exists("./log360.pth"):
        net=MyNet2()
        #net.load_state_dict(torch.load("./log360.pth", map_location='cuda:2'))
        net=torch.load("./log360.pth", 'cpu')
        print("continue training")
    else:
        net = MyNet2(num_classes=2, init_weights=True)
        net.to(device)
        print("start training anew")

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

    epochs = 2000
    #save weights path
    save_path = './pth/cc-net.pth'
    best_acc = 0.0
    train_steps = len(train_loader)

        
    trainLOSS = []  #save loss
    valACC = []     #save val acc
    
    for epoch in range(epochs):
        scheduler.step()
        print('LR:{}'.format(scheduler.get_lr()[0]))
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            #print(images.shape) [32,3,227,227]
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num

        tbwriter.add_scalar('train/loss', running_loss/train_steps, epoch)
        tbwriter.add_scalar('val/acc', val_accurate, epoch)

        trainLOSS.append(running_loss/train_steps)
        valACC.append(val_accurate)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        print(' ')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        
    npLOSS=np.array(trainLOSS)
    npVALACC=np.array(valACC)
    np.save('./loss_epoch_{}'.format(epoch), npLOSS)
    np.save('./valacc_epoch_{}'.format(epoch), npVALACC)

    print('Training Finished£¡')


if __name__ == '__main__':
    main()