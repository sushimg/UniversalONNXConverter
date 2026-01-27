import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

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

class DarknetParser(nn.Module):
    def __init__(self, cfgfile):
        super(DarknetParser, self).__init__()
        self.blocks = self._parse_cfg(cfgfile)
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
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
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

            if block['type'] == 'net':
                prev_filters = int(block.get('channels', 3))
                continue

            elif block['type'] == 'convolutional':
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

            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride == 1 and pool_size % 2:
                    module = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=pool_size // 2)
                elif stride == pool_size:
                    module = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    module = MaxPoolDark(pool_size, stride)
                
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(module)

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                module = nn.Upsample(scale_factor=stride, mode='nearest')
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                models.append(module)

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + len(models) for i in layers]
                
                total_filters = 0
                for l in layers:
                    if l < 0 or l >= len(out_filters):
                        print(f"[ERROR] Route layer {l} out of bounds")
                        total_filters += 0
                    else:
                        total_filters += out_filters[l]
                
                if 'groups' in block:
                    groups = int(block['groups'])
                    total_filters = out_filters[layers[0]] // groups

                out_filters.append(total_filters)
                prev_filters = total_filters
                out_strides.append(out_strides[layers[0]])
                models.append(nn.Identity()) 

            elif block['type'] == 'shortcut':
                out_filters.append(prev_filters)
                out_strides.append(out_strides[-1])
                models.append(nn.Identity()) 

            elif block['type'] == 'yolo':
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(nn.Identity())

            else:
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

            if block_type in ['convolutional', 'maxpool', 'upsample']:
                x = module(x)
                outputs[i] = x
            
            elif block_type == 'route':
                layers = block['layers'].split(',')
                layers = [int(l) if int(l) > 0 else int(l) + i for l in layers]
                
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
                from_layer = from_layer if from_layer > 0 else from_layer + i
                
                x1 = outputs[from_layer]
                x2 = outputs[i - 1]
                x = x1 + x2
                
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                outputs[i] = x

            elif block_type == 'yolo':
                yolo_outputs.append(x)
                outputs[i] = x
            
            else:
                outputs[i] = x
        
        if len(yolo_outputs) > 0:
            return yolo_outputs
        return x

    def load_weights(self, weightfile):
        with open(weightfile, 'rb') as fp:
            header = np.fromfile(fp, count=5, dtype=np.int32)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        total_len = weights.size
        mismatch_warned = False 

        for i, (module, block) in enumerate(zip(self.models, self.blocks[1:])):
            if block['type'] == 'convolutional':
                conv_layer = module[0]
                batch_normalize = int(block.get('batch_normalize', 0))
                
                def safe_load(param, num_el, current_ptr):
                    nonlocal mismatch_warned
                    if current_ptr + num_el > total_len:
                        if not mismatch_warned:
                            print(f"\n[WARNING] CFG and weights mismatch! (Layer: {i})")
                            print("   -> Remaining weights will be assigned RANDOM values.")
                            mismatch_warned = True
                        rand_data = torch.randn(num_el).view_as(param)
                        param.data.copy_(rand_data)
                        return current_ptr, False
                    
                    w_data = torch.from_numpy(weights[current_ptr : current_ptr + num_el]).view_as(param)
                    param.data.copy_(w_data)
                    return current_ptr + num_el, True

                if batch_normalize:
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()
                    
                    ptr, ok1 = safe_load(bn_layer.bias, num_b, ptr)
                    ptr, ok2 = safe_load(bn_layer.weight, num_b, ptr)
                    ptr, ok3 = safe_load(bn_layer.running_mean, num_b, ptr)
                    ptr, ok4 = safe_load(bn_layer.running_var, num_b, ptr)
                    
                else:
                    num_b = conv_layer.bias.numel()
                    ptr, _ = safe_load(conv_layer.bias, num_b, ptr)
                
                num_w = conv_layer.weight.numel()
                ptr, _ = safe_load(conv_layer.weight, num_w, ptr)

        print("Applying Weights Sanitize process...")
        sanitized_count = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if (m.running_var < 0).any():
                    m.running_var.data.clamp_(min=1e-5)
                    sanitized_count += 1
        if sanitized_count > 0:
            print(f"[SUCCESS] {sanitized_count} BatchNorm layers sanitized.")
