'''路径'''
test_dir = './test/'
train_dir = './train/'
label_dir = './label_data/'
save_dir = './weights/'
weight_dir = './best_weight/'
input_dir = './frontend/static/images'
output_dir = './frontend/static/images'


'''参数设置'''
num_classes = 4 #预测分类数，包括background,apple,banana,orange
iou_threshold = 0.5 #iou阈值，小于此阈值的iou不予通过
min_score = 0.6 #score阈值，小于此阈值的bbox不予通过
label_to_name = ['background', 'apple', 'banana', 'orange'] #索引到类别的映射

'''功能函数'''
# 类别到索引的映射
def class_to_idx(class_name):
        return {
            'background': 0,
            'apple': 1,
            'banana': 2,
            'orange': 3
        }[class_name]

# 求平均值
def average(list):
    average = sum(list) / len(list) if list else 0
    return average