import torch
import torch.nn as nn
import os

class RawNPUHead(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.nl = original_layer.nl
        self.cv2 = original_layer.cv2

    def forward(self, x):
        res = []
        for i in range(self.nl):
            feat = self.cv2[i](x[i])
            res.append(feat)
        return res

def export_pytorch_to_onnx(model, input_shape, output_path, opset=11):
    dummy_input = torch.randn(*input_shape, requires_grad=True)
    
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )
        return True
    except Exception as e:
        print(f"[ERROR] Generic Export Failed: {e}")
        raise e

def try_export_ultralytics(model, input_shape, output_path, opset=18):
    model_layers = None
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        model_layers = model.model
    elif isinstance(model, nn.Sequential):
        model_layers = model
    
    if model_layers is None:
        return False

    try:
        last_layer = model_layers[-1]
        if not (hasattr(last_layer, 'cv2') and hasattr(last_layer, 'nl')):
            return False
    except:
        return False

    print(f"[SUCCESS] Ultralytics structure detected. Applying NPU Head transformation...")

    try:
        new_layer = RawNPUHead(last_layer)
        
        if hasattr(last_layer, 'i'): new_layer.i = last_layer.i
        if hasattr(last_layer, 'f'): new_layer.f = last_layer.f
        if hasattr(last_layer, 'type'): new_layer.type = last_layer.type
        
        model_layers[-1] = new_layer
        print("   -> Detect layer replaced with RawNPUHead (Metadata preserved).")
        
    except Exception as e:
        print(f"[WARNING] Transformation error: {e}")
        return False

    try:
        import onnx
        
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)

        model.eval()

        print(f"   -> Starting ONNX Export (Opset: {opset})...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=['images'],
            output_names=['output0'],
            do_constant_folding=True
        )

        onnx_model = onnx.load(output_path)
        onnx.save_model(
            onnx_model, 
            output_path, 
            save_as_external_data=False 
        )
        
        data_file = output_path + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)
            print("   -> External data (.data) merged into main file.")

        return True

    except ImportError:
        print("[ERROR] 'onnx' library not found. Run `pip install onnx`.")
        raise
    except Exception as e:
        print(f"[ERROR] Ultralytics Export Failed: {e}")
        raise e
