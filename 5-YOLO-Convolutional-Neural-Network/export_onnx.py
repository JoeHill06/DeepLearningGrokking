"""
Run this once to export model1.pth → model.onnx for the docs site.

    python export_onnx.py

The output file goes directly into the docs folder where the site expects it.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Import the model definition the same way train.ipynb does
import importlib.util, types

# Load model architecture inline (mirrors model.ipynb)
class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                    padding=kernel_size // 2)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.Silu = torch.nn.SiLU()

    def forward(self, x):
        return self.Silu(self.batch_norm(self.conv(x)))


architecture = [
    [-1, 'Conv',    [32,   3, 1]],
    [-1, 'MaxPool', [2, 2]],
    [-1, 'Conv',    [64,   3, 1]],
    [-1, 'MaxPool', [2, 2]],
    [-1, 'Conv',    [128,  3, 1]],
    [-1, 'Conv',    [64,   1, 1]],
    [-1, 'Conv',    [128,  3, 1]],
    [-1, 'MaxPool', [2, 2]],
    [-1, 'Conv',    [256,  3, 1]],
    [-1, 'Conv',    [128,  1, 1]],
    [-1, 'Conv',    [256,  3, 1]],
    [-1, 'MaxPool', [2, 2]],
    [-1, 'Conv',    [512,  3, 1]],
    [-1, 'Conv',    [256,  1, 1]],
    [-1, 'Conv',    [512,  3, 1]],
    [-1, 'Conv',    [256,  1, 1]],
    [-1, 'Conv',    [512,  3, 1]],
    [-1, 'MaxPool', [2, 2]],
    [-1, 'Conv',    [1024, 3, 1]],
    [-1, 'Conv',    [512,  1, 1]],
    [-1, 'Conv',    [1024, 3, 1]],
    [-1, 'Conv',    [512,  1, 1]],
    [-1, 'Conv',    [1024, 3, 1]],
    [-1, 'Conv',    [1024, 3, 1]],
    [-1, 'Conv',    [1024, 3, 1]],
    [-1, 'Conv',    [170,  1, 1]],
]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        in_ch = 3
        for i, (_, module_type, args) in enumerate(architecture):
            is_last = (i == len(architecture) - 1)
            if module_type == 'Conv':
                out_ch, ks, st = args
                if is_last:
                    self.layers.append(torch.nn.Conv2d(in_ch, out_ch, ks, st,
                                                       padding=ks // 2))
                else:
                    self.layers.append(Conv(in_ch, out_ch, ks, st))
                in_ch = out_ch
            elif module_type == 'MaxPool':
                self.layers.append(torch.nn.MaxPool2d(*args))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


weights_path = os.path.join(os.path.dirname(__file__), 'model1.pth')
out_path = os.path.join(os.path.dirname(__file__),
                        '..', 'docs', 'networks',
                        '5-yolo-convolutional-neural-network', 'model.onnx')
out_path = os.path.normpath(out_path)

model = Model()
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

dummy = torch.zeros(1, 3, 416, 416)

torch.onnx.export(
    model, dummy, out_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17,
    dynamo=False,
)

size_mb = os.path.getsize(out_path) / 1e6
print(f"Exported → {out_path}  ({size_mb:.1f} MB)")
