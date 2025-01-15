from flask import Flask, request, render_template, url_for, send_from_directory
import os


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './frontend/static/images'  # 设置上传文件夹
app.config['PROCESSED_FOLDER'] = './frontend/static/images'  # 设置处理后的图片文件夹

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file is None:
        return 'no file error'
    if file:
        file_name = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload.jpg'))

        # 执行预测
        import subprocess
        # 使用自己的python.exe
        subprocess.run(['d:\Env\Fruit\Scripts\python.exe', 'inference.py'])
        subprocess.run(['d:\Env\Fruit\Scripts\python.exe', 'xml_show.py'])

        # 显示图片
        return render_template('display_image.html')

if __name__ == '__main__':
    app.run(debug=True)
