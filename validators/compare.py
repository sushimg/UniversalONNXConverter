import torch
import numpy as np
import onnxruntime as ort

def validate_conversion(pt_model, onnx_path, input_shape, tolerance=1e-4):
    dummy_input = torch.randn(*input_shape).float()
    
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(dummy_input)
    
    if isinstance(pt_out, (list, tuple)):
        pt_out = [x.numpy() for x in pt_out]
    else:
        pt_out = [pt_out.numpy()]
        
    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"   [ERROR] ONNX Runtime loading failed: {e}")
        return False

    input_name = session.get_inputs()[0].name
    try:
        onnx_out = session.run(None, {input_name: dummy_input.numpy()})
    except Exception as e:
        print(f"   [ERROR] ONNX Runtime execution failed: {e}")
        return False
    
    print(f"   -> Output count: PT={len(pt_out)} vs ONNX={len(onnx_out)}")
    
    if len(pt_out) != len(onnx_out):
        print(f"   [ERROR] Output count mismatch! PT has {len(pt_out)}, ONNX has {len(onnx_out)}")
        return False

    all_pass = True
    for i, (p, o) in enumerate(zip(pt_out, onnx_out)):
        if p.shape != o.shape:
            print(f"   [ERROR] Output {i} shape mismatch! PT: {p.shape}, ONNX: {o.shape}")
            all_pass = False
            continue
            
        mae = np.mean(np.abs(p - o))
        
        if np.isnan(mae):
            print(f"   [ERROR] Output {i} contains NaN values!")
            all_pass = False
        elif mae > tolerance:
            print(f"   [WARNING] Output {i} exceeded tolerance! ({mae:.6f} > {tolerance})")
            all_pass = False
        else:
            print(f"   [SUCCESS] Output {i} validated. (MAE: {mae:.6f})")
            
    return all_pass
