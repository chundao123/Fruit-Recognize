from lxml import etree
import cv2

from util import input_dir, output_dir


# 读取 xml 文件信息，并返回字典形式
def parse_xml_to_dict(xml):
    if len(xml) == 0:  # 遍历到底层，直接返回 tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


# xml 标注文件的可视化
def xmlShow(img,xml,save = True):
    image = cv2.imread(img)
    with open(xml, encoding='gb18030', errors='ignore') as fid:  # 防止出现非法字符报错
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)["annotation"]  # 读取 xml文件信息

    if 'object' not in data:
        cv2.imwrite(output_dir + '/result.png', image)
        return

    ob = []
    for i in data['object']:

        # 获取预测信息
        name = str(i['name'])
        confidence = str(i['confidence'])[0:4]
        bbox = i['bndbox']
        xmin = int(float(bbox['xmin']))
        ymin = int(float(bbox['ymin']))
        xmax = int(float(bbox['xmax']))
        ymax = int(float(bbox['ymax']))

        tmp = [name, confidence, xmin, ymin, xmax, ymax]
        ob.append(tmp)

    # 绘制预测框、预测类别、预测分数
    for name, confidence, xmin, ymin, xmax, ymax in ob:
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color=(255,0,0), thickness=2)
        cv2.putText(image, name, (xmin+40,ymin-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0,0,0))
        cv2.putText(image, confidence, (xmin,ymin-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0,0,0))

    # 保存图像
    if save:
        cv2.imwrite(output_dir + '/result.png', image)

    # 展示图像
    #cv2.imshow('test',image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path =  input_dir + '/upload.jpg'  # 传入图片

    labels_path = img_path.replace('img', 'xml')       # 自动获取对应的 xml 标注文件
    labels_path = img_path.replace('.jpg', '.xml')

    xmlShow(img=img_path, xml=labels_path,save=True)