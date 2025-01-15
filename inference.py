import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from util import weight_dir, input_dir, num_classes, min_score, label_to_name


'''图片类'''
class FruitDataset(Dataset):

    # 从指定路径读取图片
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    # 获取图片、图片名称、图宽、图高
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_folder, img_name)
        width, height = Image.open(img_path).size

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, img_name, width ,height


'''加载模型'''
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
state_dict = torch.load(weight_dir + 'loss0.007807198911905289.pth')
model.load_state_dict(state_dict, strict=False)


'''GPU'''
test_on_gpu = torch.cuda.is_available()

if not test_on_gpu:
    print("CUDA not avaliable. testing on CPU...")
else:
    print("CUDA avaliable. testing on GPU...")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)


'''数据预处理'''
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])


'''数据集'''
infer_dataset = FruitDataset(image_folder=input_dir, transform=transform)
infer_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False)


'''进行测试'''
model.eval()
with torch.no_grad():  # 禁用梯度计算
    for image, img_name, w, h in infer_loader:

        print("\n预测图片{}".format(img_name))

        # 使用模型进行预测
        image = image.to(device)
        outputs = model(image)

        # 从outputs中提取预测结果
        pred_bboxes = outputs[0]['boxes']
        pred_labels = outputs[0]['labels']
        pred_scores = outputs[0]['scores']

        bboxes_list = pred_bboxes.tolist()
        labels_list = pred_labels.tolist()
        scores_list = pred_scores.tolist()

        print("预测结果:")
        for bbox, label, score in zip(bboxes_list, labels_list, scores_list):
            print(bbox, "\t", label_to_name[label], "\t", score)

        # 创建元素：根、文件名、图宽、图高、通道数
        root = Element('annotation')
        filename = SubElement(root, 'filename')
        filename.text = img_name[0]
        size = SubElement(root, 'size')
        width = SubElement(size, 'width')
        width.text = str(w.item())
        height = SubElement(size, 'height')
        height.text = str(h.item())
        depth = SubElement(size, 'depth')
        depth.text = '3'

        for bbox, label, score in zip(bboxes_list, labels_list, scores_list):

            # 忽略score过低的框
            if score < min_score:
                continue

            # 创建object元素
            object_elem = SubElement(root, 'object')

            # 创建label元素并设置文本
            label_elem = SubElement(object_elem, 'name')
            label_elem.text = label_to_name[label]
            
            # 创建bndbox元素并设置属性
            bndbox = SubElement(object_elem, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            ymin = SubElement(bndbox, 'ymin')
            xmax = SubElement(bndbox, 'xmax')
            ymax = SubElement(bndbox, 'ymax')
            xmin.text = str(bbox[0])
            ymin.text = str(bbox[1])
            xmax.text = str(bbox[2])
            ymax.text = str(bbox[3])
                       
            # 创建confidence元素并设置文本
            confidence = SubElement(object_elem, 'confidence')
            confidence.text = str(score)

        # 将ElementTree转换成字符串
        rough_string = tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml_as_string = reparsed.toprettyxml(indent="   ")

        # 保存到文件
        with open(input_dir + '/' + img_name[0].replace('.jpg', '.xml'), "w") as f:
            f.write(pretty_xml_as_string)