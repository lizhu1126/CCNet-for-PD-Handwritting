#test.py for test set

import os,json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from cc_net_model import MyNet2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transform = transforms.Compose(
        [transforms.Resize((227, 227)),
         transforms.ToTensor(),
         transforms.Normalize([0.5],[0.5])])
         
    # load image
    img_path = "./data/test/"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    
    # read class_indict
    json_path = './class_indices_change.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = MyNet2(num_classes=2).to(device)

    # load model weights
    weights_path = "./pth/cc-net-i3.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    

    trueNum=0   #the num of pre right
    preNum=0    #the num of all pre imgs

    # 1.find folder  2.find imgs
    for homes, DIRS, FILES in os.walk(img_path):
        if homes == img_path:
            continue
        fileNames = os.listdir(homes)
        for file in fileNames:
            img_folder = homes + '/'+ file  # every img
            img = Image.open(img_folder).convert('RGB')

            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)    #[1,1,227,227]

            model.eval()
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                #predict = torch.max(output, dim=1)[1]
                predict_cla = torch.argmax(predict).numpy()

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
            true_cla = os.path.split(homes)[-1]
            #print(print_res)
            #print(class_indict[str(predict_cla)])

            if class_indict[str(predict_cla)] == (true_cla):
                trueNum=trueNum+1
            preNum=preNum+1
            print(true_cla,class_indict[str(predict_cla)])


    preACC = trueNum / preNum
    print("pre correct:{} = true:{} / all:{}".format(preACC, trueNum, preNum))
    print("pre correct:{}".format(preACC))




if __name__ == '__main__':
    main()
