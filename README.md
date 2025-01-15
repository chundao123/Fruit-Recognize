| - __pycache__ - 运行时自动产生的预编译的util
|
| - best_weight - 存放可以预测时真正使用的模型参数
|
| - fronted - | - static - | - background - 存放网页美化图
|             |            |
|             |            | - images - 存放jpg上传图、对应的xml和png结果图
|             |
|             | - templates - | - display_image.html - 显示图片的网页
|             |               |
|             |               | - upload.html - 上传图片的网页
|             |
|             | - app.py - 调用网页，保存上传的图片，调用相关函数
|
| - label_data - 标注的20张自己标注的图片jpg和对应mxl(未使用)
|
| - test - 下载的数据集，包含60张测试图片jpg和对应mxl
|
| - weights - 参数文件夹，存储训练好的模型参数
|
| - 运行截图 - 包含一些函数运行的截图
|
| - inference.py - 读取一张图片预测并保存相应xml
|
| - METRICS.docx - predic.py的输出指标
|
| - predict.py - 输入test图片并调用模型计算IoU,Precision,Recall和Mean Average Precision
|
| - README.txt - 此文件
|
| - train.py - 训练模型并保存参数
|
| - util.py - 路径，参数设置和一些功能函数
|
| - xml_show.py - 通过jpg和对应xml绘制png结果图


！！！运行步骤！！！

1：创建best_weight文件夹于根目录下,从链接 https://pan.baidu.com/s/1wbp5DAln-g0Whr24hc1SVg?pwd=jlbq 下载参数并将其存至/best_weight文件夹下

2：准备环境，安装关键包：torch, torchvision, flask（可能还用到了一些包但是忘了，请根据实际情况补下）

3：修改frontend/app.py33合34行的代码，更改里面的解释器路径为自己配置的python.exe

4：运行frontend/app.py，进入输出信息里的网址开始预测