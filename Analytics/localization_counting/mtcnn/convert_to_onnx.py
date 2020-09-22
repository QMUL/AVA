from models.mtcnn import MTCNN, PNet, RNet, ONet, prewhiten, fixed_image_standardization
from models.utils.detect_face import extract_face
import torch
import numpy as np
import cv2

import os

from tqdm import tqdm 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)

# ------------------------ export -----------------------------

print("==> Exporting model to ONNX format")
input_names = ['images']
output_names = ['scores']
inputs_pnet = torch.randn(1, 3, 649, 1153).to(device)
#inputs_pnet = torch.randn(1, 3, 460, 817).to(device)
inputs_rnet = torch.randn(1, 3, 24, 24).to(device)
inputs_onet = torch.randn(1, 3, 48, 48).to(device)

#state_dict_path = os.path.join(os.path.dirname(__file__), 'pnet.pt')
#state_dict = torch.load(state_dict_path)
#self.load_state_dict(state_dict)

#dynamic_axes = {'images': {2:'h',3:'w'}}
torch_out = torch.onnx._export(mtcnn.pnet, inputs_pnet, 'pnet.onnx', export_params=True, verbose=False, input_names=input_names, output_names=output_names, opset_version=11)
#torch_out = torch.onnx._export(mtcnn.pnet, inputs_pnet, 'pnet.onnx', dynamic_axes=dynamic_axes, export_params=True, verbose=False, input_names=input_names, output_names=output_names)
torch_out = torch.onnx._export(mtcnn.rnet, inputs_rnet, 'rnet.onnx', export_params=True, verbose=False, input_names=input_names, output_names=output_names)
torch_out = torch.onnx._export(mtcnn.onet, inputs_onet, 'onet.onnx', export_params=True, verbose=False, input_names=input_names, output_names=output_names)

