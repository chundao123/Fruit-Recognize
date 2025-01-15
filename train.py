import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from xml.etree import ElementTree as ET
from PIL import Image
import os

from util import class_to_idx, train_dir, save_dir, num_classes


'''GPU'''
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA not avaliable. training on CPU...")
else:
    print("CUDA avaliable. training on GPU...")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''数据集定义'''
class FruitDataset(Dataset):

    # 从指定路径读取图片
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    # 获取图片、预测框、预测类别
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_folder, img_name)

        annotation_path = os.path.join(self.image_folder, img_name.replace('.jpg', '.xml'))
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        bndboxes = []
        labels = []

        for obj in root.iter('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = bbox.find('xmin').text
            ymin = bbox.find('ymin').text
            xmax = bbox.find('xmax').text
            ymax = bbox.find('ymax').text
            bndboxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            labels.append(class_to_idx(label))

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        bndboxes = torch.tensor(bndboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return img, bndboxes, labels


'''数据预处理'''
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])


'''数据集'''
dataset = FruitDataset(image_folder=train_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


'''构建模型'''
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)


'''训练模块'''
model.train()
min_loss = 100
for epoch in range(20):
    print("第{}轮训练".format(epoch+1))
    train_step = 0
    for images, bndboxes, labels in data_loader:

        # 格式转化并输出
        targets = [{'boxes': bndboxes[i], 'labels': labels[i]} for i in range(len(bndboxes))]
        outputs = model(images, targets)

        # 计算loss
        loss_classifier = outputs['loss_classifier']
        loss_box_reg = outputs['loss_box_reg']
        loss_objectness = outputs['loss_objectness']
        loss_rpn_box_reg = outputs.get('loss_rpn_box_reg', 0)
        loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg

        # 学习、迭代
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1
        print("第{}轮的第{}次训练的loss:{}\t历史最小loss:{}".format((epoch+1),train_step,loss.item(),min_loss))

        # 保存loss最小的参数
        if loss.item() < min_loss:
            min_loss = loss.item()
            for f in os.listdir(save_dir):
                os.remove(save_dir + f)
            file_name = save_dir + "loss" + str(loss.item()) + '.pth'
            torch.save(model.state_dict(), file_name)
            print("文件已另存为{}".format(file_name))