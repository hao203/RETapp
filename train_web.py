"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2024/6/18 11:39
"""
import subprocess
import gradio as gr
import webbrowser
import time

def train_model(batch_size_input, world_size_input, model_input, epochs_input, blr_input,
                layer_decay_input,
                weight_decay_input, nb_classes_input, data_path_input, task_input, finetune_input,
                input_size_input, drop_path_input, device):
    # 构建命令
    command = f"""
    python  main_finetune.py \
    --batch_size {batch_size_input if batch_size_input is not None else 16} \
    --world_size {world_size_input if world_size_input is not None else 1} \
    --model {model_input if model_input is not None else 'vit_large_patch16'} \
    --epochs {epochs_input if epochs_input is not None else 50} \
    --blr {blr_input if blr_input is not None else 5e-3} \
    --layer_decay {layer_decay_input if layer_decay_input is not None else 0.65}  \
    --weight_decay {weight_decay_input if weight_decay_input is not None else 0.05} \
    --drop_path {drop_path_input if drop_path_input is not None else 0.1} \
    --nb_classes {nb_classes_input if nb_classes_input is not None else 1000} \
    --data_path {data_path_input if data_path_input is not None else './data/OCTID/'} \
    --task {task_input if task_input is not None else 'finetune_OCTID/'} \
    --finetune {finetune_input if finetune_input is not None else './models/RETFound_cfp_weights.pth'} \
    --input_size {input_size_input if input_size_input is not None else 224} 
    --device {device if device is not None else 'cpu'} 
    """
    print(command)
    # 执行命令
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stdout, stderr = process.communicate()
    log_dir = "./output_dir"
    cmd = f"tensorboard --logdir={log_dir+task_input}"  # 替换为你的 TensorBoard 日志目录
    subprocess.Popen(cmd, shell=True)
    time.sleep(2)
    webbrowser.open_new_tab('http://localhost:6006/')
    # 返回命令的输出结果
    return process.stdout.read().decode()


# 创建输入组件
# batch_size_input = gr.Number(label="Batch Size")
# train_data_input = gr.Textbox(label="Train Data")
# model_input = gr.Textbox(label="Train model")
# epochs_input = gr.Number(label="Epoch")
# blr_input = gr.Number(label="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
# layer_decay_input = gr.Number(label="layer-wise lr decay from ELECTRA/BEiT")
# weight_decay_input = gr.Number(label="weight decay (default: 0.05)")
# nb_classes_input = gr.Number(label="number of the classification types")
# data_path = gr.Textbox(label="dataset path")
# task = gr.Textbox(label="dataset path")
# finetune = gr.Textbox(label="inetune from checkpoint")
# input_size = gr.Number(label="images input size")
#
# # 创建输出组件
# output_text = gr.Textbox(label="Command Output")
#
# # 创建界面
# interface = gr.Interface(fn=train_model, inputs=[batch_size_input, train_data_input], outputs=output_text,
#                          title="模型训练界面")
#
# # 启动界面
# interface.launch()

if __name__ == '__main__':
    demo = gr.Blocks(title='', theme=gr.themes.Glass())
    with demo:
        with gr.Row():
            # 创建输入组件
            batch_size_input = gr.Number(label="Batch Size")
            world_size_input = gr.Number(label="world_size")
            model_input = gr.Textbox(label="Train model")
            epochs_input = gr.Number(label="Epoch")
            blr_input = gr.Number(label="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
            layer_decay_input = gr.Number(label="layer-wise lr decay from ELECTRA/BEiT")
            weight_decay_input = gr.Number(label="weight decay (default: 0.05)")
            nb_classes_input = gr.Number(label="number of the classification types")
            data_path_input = gr.Textbox(label="dataset path")
            task_input = gr.Textbox(label="task path")
            finetune_input = gr.Textbox(label="finetune from checkpoint")
            input_size_input = gr.Number(label="images input size")
            drop_path_input = gr.Number(label="Drop path rate (default: 0.1)")
            device_input = gr.Textbox(label="device")

            output_text = gr.Textbox(label="Command Output")

            btn = gr.Button(value="Run")
            btn.click(train_model,
                      [batch_size_input, world_size_input, model_input, epochs_input, blr_input,
                       layer_decay_input, weight_decay_input, nb_classes_input, data_path_input, task_input,
                       finetune_input, input_size_input, drop_path_input, device_input], [output_text])

    demo.launch()
