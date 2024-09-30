"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2024/6/12 15:10
"""

import cv2

# 读取图片
# image = cv2.imread('/Users/houcg/PycharmProjects/pythonProject/RETFound_MAE-main/data/Retina/test/anormal/NL_001.png')
# image = cv2.imread('/Users/houcg/PycharmProjects/pythonProject/RETFound_MAE-main/data/tmp/tmp.png')
#
# # 获取图像的通道数
# channels = image.shape[2]
# print(channels)
# # 判断通道顺序
# if channels == 3:
#     print("图像处于 RGB 颜色空间")
# elif channels == 4:
#     print("图像处于 RGBA 颜色空间")
# else:
#     print("图像通道数为", channels)

import gradio as gr
import subprocess


def train_model(batch_size, train_data):
    # 构建命令
    command = f"python xxx.py --train_data {train_data} --batch_size {batch_size}"

    # 执行命令
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # 返回命令的输出结果
    return stdout.decode()


# 创建输入组件
batch_size_input = gr.Number(label="Batch Size")
train_data_input = gr.Textbox(label="Train Data")

# 创建输出组件
output_text = gr.Textbox(label="Command Output")

# 创建界面
interface = gr.Interface(fn=train_model, inputs=[batch_size_input, train_data_input], outputs=output_text,
                         title="模型训练界面")

# 启动界面
interface.launch()