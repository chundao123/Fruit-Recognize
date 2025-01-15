## 此项目为大二时人工智能原理课程作业

## 项目结构

### _pycache
运行时自动产生的预编译文件

### fronted
前端相关文件，其中：
static存放图片和结果xml
templates存放所用网页
app.py为控制网页运行的代码

### label_data
队友标注的20张图片和对应xml

### test.rar
数据集的压缩包

### inference.py
读取一张图片预测并保存结果xml

### predict.py
输入test图片并调用模型计算IoU,Precision,Recall和Mean Average Precision

### train.py
训练模型并保存参数

### util.py
路径、参数设置和一些功能函数

### xml_show.py
通过jpg和xml绘制png结果图

## 注意事项
运行前请：建立weights文件夹用于存放模型；建立best_weight文件夹并将预测用的模型存入其中；修改frontend/app.py33合34行的代码，更改里面的解释器路径为自己配置的python
