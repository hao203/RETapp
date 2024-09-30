"""
-*- coding: utf-8 -*-

@Author  : houcg
@Time    : 2024/6/5 17:33
"""
import argparse

import cv2
import torch
import gradio as gr
import models_vit
import torch.nn as nn
from torchvision import datasets
from PIL import Image
from timm.models.layers import trunc_normal_

import os

from util.datasets import build_transform
from util.pos_embed import interpolate_pos_embed

os.environ["OMP_NUM_THREADS"] = "1"

data_path = os.path.abspath("./data")
model_path = os.path.abspath("./models")
tmp_data_path = os.path.abspath("./data/tmp")
model_map = {"oct": os.path.join(model_path, "oct-checkpoint-best.pth"),
             "retina": os.path.join(model_path, "RETFound_cfp_weights.pth")}

dateset_map = {"oct": os.path.join(data_path, "OCTID"),
               "retina": os.path.join(data_path, "Retina")}
data_list = os.listdir(data_path)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    parser.add_argument('--input_size', default=224, type=int, required=False,
                        help='images input size')
    parser.add_argument('--data_path', default=dateset_map["oct"], type=str,
                        required=False,
                        help='dataset path')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', required=False,
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--device', default='cuda', required=False,
                        help='device to use for training / testing')

    parser.add_argument('--global_pool', action='store_true', required=False, )
    parser.set_defaults(global_pool=True)

    return parser


# {'anormal': 0, 'bcataract': 1, 'cglaucoma': 2, 'ddretina_disease': 3}
def build_train_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def build_pre_dataset(is_train, image_path):
    transform = build_transform(is_train, args)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image


def load_model(nb_classes, args, model_name='vit_large_patch16', best_model_name='oct'):
    model_vit = models_vit.__dict__[model_name](
        img_size=args.input_size,
        num_classes=nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # checkpoint = torch.load("./models/RETFound_cfp_weights.pth", map_location='cpu')
    checkpoint = torch.load("./models/RETFound_oct_weights.pth", map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % model_map[best_model_name])
    checkpoint_model = checkpoint['model']
    state_dict = model_vit.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    # interpolate position embedding
    interpolate_pos_embed(model_vit, checkpoint_model)
    
    # load pre-trained model
    msg = model_vit.load_state_dict(checkpoint_model, strict=False)
    # print(msg)
    
    # manually initialize fc layer
    trunc_normal_(model_vit.head.weight, std=2e-5)
    return model_vit, best_model_name


args = get_args_parser()
args = args.parse_args()
train_dateset = build_train_dataset("val", args)
id2label = {v: k for k, v in train_dateset.class_to_idx.items()}
nb_classes = len(train_dateset.classes)
print(id2label)

model, cur_model = load_model(nb_classes, args)
model.eval()

def predict(image, best_model_select):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join(tmp_data_path, f"tmp.png"), image_rgb)

    images = build_pre_dataset("val", os.path.join(tmp_data_path, f"tmp.png"))  # [1,3,224,224]
    # if cur_model != best_model_select:
    #     args.data_path = dateset_map[best_model_select]
    #     train_dateset = build_train_dataset("val", args)
    #     id2label = {v: k for k, v in train_dateset.class_to_idx.items()}
    #     nb_classes = len(train_dateset.classes)
    #
    #     model, cur_model = load_model(nb_classes, args)
    output = model(images)
    prediction_softmax = nn.Softmax(dim=1)(output)
    _, prediction_decode = torch.max(prediction_softmax, 1)
    print(prediction_decode.item())
    print(id2label[prediction_decode.item()])
    return "预测结果：" + id2label[prediction_decode.item()]


# predict(None, None)
if __name__ == '__main__':
    with gr.Blocks() as predict_page:
        image_input = gr.Image(label="上传图片", sources=['upload'], image_mode='RGB')
        best_model_select = gr.Dropdown(choices=list(model_map.keys()), label="请选择模型")
        output_text = gr.Textbox(label="结果", lines=30, max_lines=30)

        gr.Interface(fn=predict,
                     inputs=[image_input, best_model_select],
                     outputs=[output_text], )

    tabbed_interface = gr.TabbedInterface(
        [predict_page],
        [""])

    tabbed_interface.launch()
