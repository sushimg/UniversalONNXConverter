import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import warnings
from .logger import log_info, log_warning, log_success, log_error

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1

        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3

        x = F.max_pool2d(F.pad(x, (padding3, padding4, padding1, padding2), mode='replicate'),
                         self.size, stride=self.stride)
        return x

class YOLOLayer(nn.Module):
    warned = False

    def __init__(self, anchors, mask, classes, img_size, no_yolo_layer=False):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.FloatTensor([anchors[i] for i in mask])
        self.classes = classes
        self.img_size = img_size  # (height, width)
        self.no_yolo_layer = no_yolo_layer
        self.num_anchors = len(mask)
        self.register_buffer('anchor_grid', self.anchors.clone().view(1, self.num_anchors, 1, 1, 2))
        self.grid = None

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        if self.no_yolo_layer:
            return x

        if not YOLOLayer.warned:
            log_warning("This model is optimized for static shapes only. Dynamic input sizes are not supported and may lead to unexpected results.")
            YOLOLayer.warned = True

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

            B, C, H, W = x.shape
            x = x.view(B, self.num_anchors, 5 + self.classes, H, W).permute(0, 1, 3, 4, 2).contiguous()

            # Split to avoid Gather/ScatterND
            xy, wh, conf_cls = torch.split(x, [2, 2, self.classes + 1], dim=-1)

            # Activation functions
            xy = torch.sigmoid(xy)
            conf_cls = torch.sigmoid(conf_cls)

            if self.grid is None or self.grid.shape[2:4] != (H, W):
                self.grid = self._make_grid(W, H).to(x.device)

            # Decoding arithmetic
            stride_x = self.img_size[1] / W
            stride_y = self.img_size[0] / H
            
            stride = torch.tensor([stride_x, stride_y], device=x.device, dtype=x.dtype)
            xy = (xy + self.grid) * stride
            wh = torch.exp(wh) * self.anchor_grid

            pred = torch.cat((xy, wh, conf_cls), dim=-1)
            return pred.view(B, -1, 5 + self.classes)

class DarknetParser(nn.Module):
    def __init__(self, cfgfile, no_yolo_layer=False):
        super(DarknetParser, self).__init__()
        self.no_yolo_layer = no_yolo_layer
        self.blocks = self._parse_cfg(cfgfile)

        while self.blocks and self.blocks[-1]['type'] == 'contrastive':
            log_info("Pruning 'contrastive' layer found at the end of the network.")
            self.blocks.pop()

        self.net_info = self.blocks[0]
        self.width = int(self.net_info.get('width', 416))
        self.height = int(self.net_info.get('height', 416))
        self.models = self._create_network(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def _parse_cfg(self, cfgfile):
        with open(cfgfile, 'r') as file:
            lines = file.read().split('\n')
            lines = [x for x in lines if x and not x.startswith('#')]
            lines = [x.rstrip().lstrip() for x in lines]

        block = {}
        blocks = []
        for line in lines:
            if line.startswith('['):
                if block:
                    blocks.append(block)
                    block = {}
                block['type'] = line[1:-1].rstrip()
            else:
                try:
                    key, value = line.split("=")
                    block[key.rstrip()] = value.lstrip()
                except:
                    log_warning(f"Skipping malformed line in CFG: {line}")
                    continue
        blocks.append(block)
        return blocks

    def _create_network(self, blocks):
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0

        for block in blocks:
            module = nn.Sequential()
            block_type = block['type']

            if block_type == 'net':
                prev_filters = int(block.get('channels', 3))
                continue

            elif block_type == 'convolutional':
                conv_id += 1
                batch_normalize = int(block.get('batch_normalize', 0))
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block.get('pad', 0))
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']

                if batch_normalize:
                    module.add_module(f'conv{conv_id}', nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    module.add_module(f'bn{conv_id}', nn.BatchNorm2d(filters))
                else:
                    module.add_module(f'conv{conv_id}', nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))

                if activation == 'leaky':
                    module.add_module(f'leaky{conv_id}', nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'mish':
                    module.add_module(f'mish{conv_id}', Mish())
                elif activation == 'linear':
                    pass

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(module)

            elif block_type == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride == 1 and pool_size % 2:
                    module = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=pool_size // 2)
                elif stride == pool_size:
                    module = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    module = MaxPoolDark(pool_size, stride)

                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(module)

            elif block_type == 'avgpool':
                module = nn.AdaptiveAvgPool2d((1, 1))
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(module)

            elif block_type == 'local_avgpool':
                pool_size = int(block.get('size', 2))
                stride = int(block.get('stride', 1))
                module = nn.AvgPool2d(kernel_size=pool_size, stride=stride, padding=0, count_include_pad=False)

                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(module)

            elif block_type == 'upsample':
                stride = int(block['stride'])
                module = nn.Upsample(scale_factor=stride, mode='nearest')
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                models.append(module)

            elif block_type == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) >= 0 else int(i) + len(models) for i in layers]

                if 'groups' in block:
                    groups = int(block['groups'])
                    total_filters = out_filters[layers[0]] // groups
                else:
                    total_filters = 0
                    for l in layers:
                        try:
                            total_filters += out_filters[l]
                        except IndexError:
                             log_error(f"DEBUG ERROR: Block {len(models)}, Type {block_type}")
                             log_error(f"Refers to l={l}, out_filters len={len(out_filters)}")
                             log_error(f"Original layers: {block['layers']}")
                             log_error(f"Calculated layers: {layers}")
                             raise

                out_filters.append(total_filters)
                prev_filters = total_filters
                out_strides.append(out_strides[layers[0]])
                models.append(nn.Identity())

            elif block_type == 'shortcut':
                out_filters.append(prev_filters)
                out_strides.append(out_strides[-1])
                models.append(nn.Identity())

            elif block_type == 'yolo':
                mask = block['mask'].split(',')
                mask = [int(x) for x in mask]
                anchors = block['anchors'].split(',')
                anchors = [int(x) for x in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
                classes = int(block['classes'])
                img_size = (self.height, self.width)

                module = YOLOLayer(anchors, mask, classes, img_size, self.no_yolo_layer)
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(module)

            elif block_type == 'contrastive':
                log_info("Replacing middle 'contrastive' layer with Identity.")
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(nn.Identity())

            else:
                log_warning(f"Unknown block type: {block_type}. Using Identity.")
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(nn.Identity())

        return models

    def forward(self, x):
        outputs = {}
        yolo_outputs = []

        for i, block in enumerate(self.blocks[1:]):
            module = self.models[i]
            block_type = block['type']

            if block_type in ['convolutional', 'maxpool', 'upsample', 'local_avgpool', 'avgpool']:
                x = module(x)
                outputs[i] = x

            elif block_type == 'route':
                layers = block['layers'].split(',')
                layers = [int(l) if int(l) >= 0 else int(l) + i for l in layers]

                if len(layers) == 1:
                    if 'groups' in block:
                        groups = int(block['groups'])
                        group_id = int(block['group_id'])
                        val = outputs[layers[0]]
                        channels = val.shape[1]
                        start = (channels // groups) * group_id
                        end = (channels // groups) * (group_id + 1)
                        x = val[:, start:end, :, :]
                    else:
                        x = outputs[layers[0]]
                elif len(layers) > 1:
                    maps = [outputs[l] for l in layers]
                    x = torch.cat(maps, 1)
                outputs[i] = x

            elif block_type == 'shortcut':
                from_layer = int(block['from'])
                activation = block.get('activation', 'linear')
                from_layer = from_layer if from_layer >= 0 else from_layer + i

                x1 = outputs[from_layer]
                x2 = outputs[i - 1]
                x = x1 + x2

                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                outputs[i] = x

            elif block_type == 'yolo':
                x = module(x)
                yolo_outputs.append(x)
                outputs[i] = x

            else:
                outputs[i] = x

        if len(yolo_outputs) > 0:
            if self.no_yolo_layer:
                return yolo_outputs
            return torch.cat(yolo_outputs, 1)
        return x

    def load_weights(self, weightfile):
        if not weightfile or not os.path.exists(weightfile):
            log_warning(f"Weight file not found: {weightfile}")
            return

        log_info(f"Loading weights from {weightfile}")
        with open(weightfile, 'rb') as fp:
            header = np.fromfile(fp, count=5, dtype=np.int32)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        total_len = weights.size

        for i, (module, block) in enumerate(zip(self.models, self.blocks[1:])):
            if block['type'] == 'contrastive':
                log_info("Reached 'contrastive' layer during weight loading. Stopping further loading.")
                break

            if block['type'] == 'convolutional':
                conv_layer = module[0]
                batch_normalize = int(block.get('batch_normalize', 0))

                def load_tensor(param, ptr):
                    numel = param.numel()
                    if ptr + numel > total_len:
                        raise RuntimeError(f"Weight mismatch at layer {i}: expected {numel} more values, but reached EOF. "
                                           f"Check if you are using the correct weights for this .cfg file.")

                    w_data = torch.from_numpy(weights[ptr : ptr + numel]).view_as(param)
                    param.data.copy_(w_data)
                    return ptr + numel

                if batch_normalize:
                    bn_layer = module[1]
                    ptr = load_tensor(bn_layer.bias, ptr)
                    ptr = load_tensor(bn_layer.weight, ptr)
                    ptr = load_tensor(bn_layer.running_mean, ptr)
                    ptr = load_tensor(bn_layer.running_var, ptr)
                else:
                    ptr = load_tensor(conv_layer.bias, ptr)

                ptr = load_tensor(conv_layer.weight, ptr)

        log_success("Weights loaded successfully.")