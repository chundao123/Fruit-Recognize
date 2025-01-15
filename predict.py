import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from xml.etree import ElementTree as ET
from PIL import Image
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from util import average, class_to_idx, weight_dir, test_dir, num_classes, iou_threshold, min_score, label_to_name


'''数据定义'''
class FruitDataset(Dataset):

    # 从指定路径读取图片
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    # 获取图片、预测框、预测类别、图片名称
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

        return img, bndboxes, labels, img_name


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
test_dataset = FruitDataset(image_folder=test_dir, transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


'''评估指标'''
eval_all = {'ious': [], 'precisions': [], 'recalls': [], 'aps': []}
eval_apple = {'ious': [], 'precisions': [], 'recalls': [], 'aps': []}
eval_banana = {'ious': [], 'precisions': [], 'recalls': [], 'aps': []}
eval_orange = {'ious': [], 'precisions': [], 'recalls': [], 'aps': []}


'''指标计算函数'''
# 计算一个预测框和所有真实框的最大IoU
def calculate_iou(pred_bbox, true_bboxes):

    max_iou = 0
    for true_bbox in true_bboxes:
        inter_xmin = max(pred_bbox[0], true_bbox[0])
        inter_ymin = max(pred_bbox[1], true_bbox[1])
        inter_xmax = min(pred_bbox[2], true_bbox[2])
        inter_ymax = min(pred_bbox[3], true_bbox[3])

        # 计算交集的宽和高
        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)

        # 计算交集面积
        inter_area = inter_w * inter_h

        # 计算预测框和真实框的面积
        pred_box_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
        true_box_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])

        # 计算并集面积
        union_area = pred_box_area + true_box_area - inter_area

        # 计算IoU
        iou = (inter_area / union_area).item()
        max_iou = max(iou, max_iou)

    return max_iou

#计算四项指标
def calculate(true_bboxes, true_labels, pred_bboxes, pred_labels, pred_scores, iou_threshold=0.5):

    if true_bboxes.numel() == 0:
        return -1, -1, -1, -1

    # 初始化真正例和假正例计数器
    true_positives = 0
    false_positives = 0

    # 将预测框按置信度得分降序排序，并获取排序后的索引
    pred_indices = torch.argsort(pred_scores, descending=True)

    # 标记每个真实框是否已经匹配（初始化为未匹配）
    matched_true = torch.zeros(true_bboxes.shape[0], dtype=torch.bool)

    # 遍历每个预测框
    iou_list = []
    ap = 0
    recall_pre = 0
    for pred_index in pred_indices:
        pred_bbox = pred_bboxes[pred_index]
        pred_score = pred_scores[pred_index]

        # 寻找最佳匹配的真实框
        best_iou = 0
        best_true_index = -1
        for true_index in range(true_bboxes.shape[0]):
            if matched_true[true_index]:
                continue
            iou = calculate_iou(pred_bbox, true_bboxes[true_index])
            iou_list.append(iou)

            if iou > best_iou and true_labels[0][true_index] == pred_labels[pred_index.item()]:
                best_iou = iou
                best_true_index = true_index

        if best_iou >= iou_threshold and pred_score.item() > min_score:
            true_positives += 1     # 如果找到匹配的真实框，并且IoU满足阈值，则计数真正例
            matched_true[best_true_index] = True
        else:
            false_positives += 1    # 如果没有匹配的真实框或IoU不满足阈值，则计数假正例

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / true_bboxes.shape[0] if true_bboxes.shape[0] > 0 else 0
        ap += precision * (recall - recall_pre)
        recall_pre = recall
    
    iou = average(iou_list)

    return iou, precision, recall, ap


'''进行测试'''
model.eval()
with torch.no_grad():  # 禁用梯度计算
    ap_data = {'precisions': [], 'recalls': []}
    for images, true_bboxes, true_labels, img_name in test_data_loader:

        print("\n评估图片{}".format(img_name))

        # 使用模型进行预测
        images = images.to(device)
        true_bboxes = true_bboxes.to(device)
        true_labels = true_labels.to(device)
        outputs = model(images)

        # 从outputs中提取预测结果
        pred_bboxes = outputs[0]['boxes']
        pred_labels = outputs[0]['labels']
        pred_scores = outputs[0]['scores']

        # 构建水果数据
        apple_bboxes = torch.empty(0)
        apple_labels = torch.empty(0)
        banana_bboxes = torch.empty(0)
        banana_labels = torch.empty(0)
        orange_bboxes = torch.empty(0)
        orange_labels = torch.empty(0)

        # 添加水果数据
        for i in range(true_labels.size(0)):
            if pred_labels[i] == torch.tensor(1):
                apple_bboxes = torch.cat((apple_bboxes, true_bboxes[i].unsqueeze(0)))
                apple_labels = torch.cat((apple_labels, true_labels[i].unsqueeze(0)))
            
            elif pred_labels[i] == torch.tensor(2):
                banana_bboxes = torch.cat((banana_bboxes, true_bboxes[i].unsqueeze(0)))
                banana_labels = torch.cat((banana_labels, true_labels[i].unsqueeze(0)))
            
            elif pred_labels[i] == torch.tensor(3):
                orange_bboxes = torch.cat((orange_bboxes, true_bboxes[i].unsqueeze(0)))
                orange_labels = torch.cat((orange_labels, true_labels[i].unsqueeze(0)))

        print("预测结果:")
        for bbox, label, score in zip(pred_bboxes.tolist(), pred_labels.tolist(), pred_scores.tolist()):
            print(bbox, "\t", label_to_name[label], "\t", score)

        print("真实Apple数据:")
        if apple_bboxes.size(0) != 0:
            for bbox, label in zip(apple_bboxes[0].tolist(), apple_labels[0].tolist()):
                print(bbox, "\t", label_to_name[int(label)], "\t", score)

        print("真实Banana数据:")
        if banana_bboxes.size(0) != 0:
            for bbox, label in zip(banana_bboxes[0].tolist(), banana_labels[0].tolist()):
                print(bbox, "\t", label_to_name[int(label)], "\t", score)

        print("真实Orange数据:")
        if orange_bboxes.size(0) != 0:
            for bbox, label in zip(orange_bboxes[0].tolist(), orange_labels[0].tolist()):
                print(bbox, "\t", label_to_name[int(label)], "\t", score)

        # 计算all指标
        iou, precision, recall, ap = calculate(true_bboxes, true_labels, pred_bboxes, pred_labels, pred_scores, iou_threshold)
        print("all_iou=", iou, "\tall_precision=", precision, "\tall_recall=", recall, "\tall_ap=", ap)
        eval_all['ious'].append(iou)
        eval_all['precisions'].append(precision)
        eval_all['recalls'].append(recall)
        eval_all['aps'].append(ap)

        # 计算apple指标
        iou, precision, recall, ap = calculate(apple_bboxes, apple_labels, pred_bboxes, pred_labels, pred_scores, iou_threshold)
        if iou != -1:
            print("apple_iou=", iou, "\tapple_precision=", precision, "\tapple_recall=", recall, "\tapple_ap=", ap)
            eval_apple['ious'].append(iou)
            eval_apple['precisions'].append(precision)
            eval_apple['recalls'].append(recall)
            eval_apple['aps'].append(ap)
        else:
            print("本图没有apple元素，不进行apple相关指标计算")

        # 计算banana指标
        iou, precision, recall, ap = calculate(banana_bboxes, banana_labels, pred_bboxes, pred_labels, pred_scores, iou_threshold)
        if iou != -1:
            print("banana_iou=", iou, "\tbanana_precision=", precision, "\tbanana_recall=", recall, "\tbanana_ap=", ap)
            eval_banana['ious'].append(iou)
            eval_banana['precisions'].append(precision)
            eval_banana['recalls'].append(recall)
            eval_banana['aps'].append(ap)
        else:
            print("本图没有banana元素，不进行banana相关指标计算")

        # 计算orange指标
        iou, precision, recall, ap = calculate(orange_bboxes, orange_labels, pred_bboxes, pred_labels, pred_scores, iou_threshold)
        if iou != -1:
            print("orange_iou=", iou, "\torange_precision=", precision, "\torange_recall=", recall, "\torange_ap=", ap)
            eval_orange['ious'].append(iou)
            eval_orange['precisions'].append(precision)
            eval_orange['recalls'].append(recall)
            eval_orange['aps'].append(ap)
        else:
            print("本图没有orange元素，不进行orange相关指标计算")


# 处理打印评估结果
iou_all = average(eval_all['ious'])
precision_all = average(eval_all['precisions'])
recall_all = average(eval_all['recalls'])
map_all = average(eval_all['aps'])
iou_apple = average(eval_apple['ious'])
precision_apple = average(eval_apple['precisions'])
recall_apple = average(eval_apple['recalls'])
map_apple = average(eval_apple['aps'])
iou_banana = average(eval_banana['ious'])
precision_banana = average(eval_banana['precisions'])
recall_banana = average(eval_banana['recalls'])
map_banana = average(eval_banana['aps'])
iou_orange = average(eval_orange['ious'])
precision_orange = average(eval_orange['precisions'])
recall_orange = average(eval_orange['recalls'])
map_orange = average(eval_orange['aps'])

print("-------------------------------------------------")
print("最终结果：")
print("{:<20}{:<20}{:<20}{:<20}{:<20}".format("Class", "IoU", "Precision", "Recall", "Mean Average Precision"))
print("-------------------------------------------------")
print("{:<20}{:<20}{:<20}{:<20}{:<20}".format("All", iou_all, precision_all, recall_all, map_all))
print("-------------------------------------------------")
print("{:<20}{:<20}{:<20}{:<20}{:<20}".format("Apple", iou_apple, precision_apple, recall_apple, map_apple))
print("-------------------------------------------------")
print("{:<20}{:<20}{:<20}{:<20}{:<20}".format("Banana", iou_banana, precision_banana, recall_banana, map_banana))
print("-------------------------------------------------")
print("{:<20}{:<20}{:<20}{:<20}{:<20}".format("Orange", iou_orange, precision_orange, recall_orange, map_orange))