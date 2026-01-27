import onnx
import os

def optimize_for_npu(onnx_path, target_ir=8, simplify=True):
    model = onnx.load(onnx_path)
    
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
            print("   -> Running onnx-simplifier...")
            model, check = onnx_simplify(model)
            if not check:
                print("   [WARNING] Simplifier could not validate the model, using original.")
        except ImportError:
            print("   [WARNING] onnx-simplifier not installed, skipping. (pip install onnx-simplifier)")
    
    if model.ir_version > target_ir:
        print(f"   -> Reducing IR Version: v{model.ir_version} -> v{target_ir}")
        model.ir_version = target_ir

    onnx.save(model, onnx_path)
    
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    
    return onnx_path
