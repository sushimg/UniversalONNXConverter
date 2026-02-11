import argparse
import os
import sys
import torch

from core.darknet_parser import DarknetParser
from exporters.engine import export_pytorch_to_onnx, try_export_ultralytics
from exporters.post_process import optimize_for_npu
from validators.compare import validate_conversion
from validators.inference import validate_with_opencv
from core.logger import log_info, log_warning, log_error, log_success, set_color_mode

def show_tutorial():
    tutorial_path = os.path.join(os.path.dirname(__file__), 'tutorial.txt')
    if os.path.exists(tutorial_path):
        with open(tutorial_path, 'r') as f:
            print(f.read())
    else:
        log_error("Tutorial file not found.")

def get_args():
    parser = argparse.ArgumentParser(
        description="Universal ONNX Converter: Optimize models for NPU deployment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    group = parser.add_argument_group('Required Arguments')
    group.add_argument('--mode', type=str, choices=['darknet', 'pytorch'], 
                        help='Input model type:\n'
                             '  darknet: Use .cfg and .weights\n'
                             '  pytorch: Use .pt file')
    
    group = parser.add_argument_group('Configuration')
    group.add_argument('--weights', type=str, help='Path to weight file (.weights or .pt)')
    group.add_argument('--cfg', type=str, help='Path to .cfg file (Darknet mode only)')
    group.add_argument('--output', type=str, default='model.onnx', help='Output path (default: model.onnx)')
    group.add_argument('--output-mode', type=str, default='onnx', choices=['onnx', 'pytorch'],
                        help='Output format:\n'
                             '  onnx: NPU optimized Opset 11\n'
                             '  pytorch: Standard PyTorch file')
    group.add_argument('--shape', type=int, nargs=4, default=None, 
                        help='Input shape: B C H W (default: 1 3 416 416)')
    
    group = parser.add_argument_group('Advanced Flags')
    group.add_argument('--validate', action='store_true', help='Compare PyTorch and ONNX outputs')
    group.add_argument('--no-simplify', action='store_true', help='Skip ONNX optimization step')
    group.add_argument('--no-yolo-layer', action='store_true', help='Output raw tensors instead of decoded YOLO predictions')
    group.add_argument('--tutorial', action='store_true', help='Show usage examples and exit')

    return parser.parse_args()

def load_model(args):
    log_info(f"Loading {args.mode.upper()} model...")
    model = None
    
    if args.mode == 'darknet':
        if not args.cfg:
            log_error("Darknet mode requires a --cfg file.")
            sys.exit(1)
        try:
            model = DarknetParser(args.cfg, no_yolo_layer=args.no_yolo_layer)
            if args.weights:
                model.load_weights(args.weights)
            else:
                log_info("No weights provided. Using random initialization.")
            log_success("Darknet model is ready.")
        except Exception as e:
            log_error(f"Could not load Darknet model: {e}")
            sys.exit(1)

    elif args.mode == 'pytorch':
        if not args.weights:
            log_error("PyTorch mode requires a --weights file.")
            sys.exit(1)
            
        log_info(f"Reading file: {args.weights}")
        try:
            try:
                loaded = torch.load(args.weights, map_location='cpu', weights_only=False)
            except TypeError:
                loaded = torch.load(args.weights, map_location='cpu')

            if isinstance(loaded, dict):
                if 'model' in loaded:
                    model = loaded['model']
                    if hasattr(model, 'float'): model.float()
                elif 'state_dict' in loaded:
                    log_error("Found only state_dict. Model architecture is missing.")
                    sys.exit(1)
                else:
                    model = loaded
            else:
                model = loaded
        except ModuleNotFoundError as e:
            if 'models' in str(e) or 'utils' in str(e):
                log_info("YOLOv5 model detected but local 'models' module missing.")
                log_info("Attempting to load via torch.hub (this may download dependencies)...")
                try:
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights, force_reload=False, trust_repo=True)
                    log_success("Model successfully loaded using torch.hub.")
                except Exception as hub_e:
                    log_error(f"Failed to load via torch.hub: {hub_e}")
                    log_error("Please ensure you are running the command in the official YOLOv5 directory.")
                    sys.exit(1)
            else:
                log_error(f"Module not found while loading model: {e}")
                sys.exit(1)
        except Exception as e:
            log_error(f"Could not load PyTorch model: {e}")
            sys.exit(1)

        if model is not None and hasattr(model, 'fuse'):
            try: model.fuse()
            except: pass
            
        log_success("PyTorch model is ready.")

    if hasattr(model, 'eval'):
        model.eval()
    return model

def main():
    set_color_mode()
    args = get_args()
    
    if args.tutorial:
        show_tutorial()
        return

    if not args.mode:
        log_error("Please specify --mode or use --tutorial.")
        return

    if args.no_yolo_layer and args.mode != 'darknet':
        log_warning("--no-yolo-layer is only applicable in darknet mode. Ignoring this flag.")
        args.no_yolo_layer = False

    model = load_model(args)

    if args.shape is None:
        if args.mode == 'darknet' and hasattr(model, 'width') and hasattr(model, 'height'):
            args.shape = [1, 3, model.height, model.width]
            log_info(f"Auto-detected input shape from CFG: {args.shape}")
        else:
            args.shape = [1, 3, 416, 416]
            log_info(f"Using default input shape: {args.shape}")
    else:
        if args.mode == 'darknet' and hasattr(model, 'width') and hasattr(model, 'height'):
            cfg_shape = [1, 3, model.height, model.width]
            if args.shape != cfg_shape:
                log_warning(f"Provided shape {args.shape} mismatch with CFG {cfg_shape}. Using provided shape.")
            else:
                log_info(f"Using input shape: {args.shape}")
        else:
            log_info(f"Using input shape: {args.shape}")

    if args.output_mode == 'pytorch':
        log_info("Saving model to PyTorch format...")
        save_path = args.output
        if not save_path.lower().endswith('.pt'):
            save_path += '.pt'
        if args.validate:
            log_info("Validation is currently only supported for ONNX output mode. Skipping validation.")
        torch.save(model, save_path)
        log_success(f"Model saved to {save_path}")

    elif args.output_mode == 'onnx':
        log_info("Starting ONNX export process...")
        save_path = args.output
        if not save_path.lower().endswith('.onnx'):
            save_path += '.onnx'

        try:
            is_ultralytics = try_export_ultralytics(model, args.shape, save_path)
            
            if not is_ultralytics:
                log_info("Standard model detected. Using generic export.")
                export_pytorch_to_onnx(model, args.shape, save_path)
            
            log_info("Running NPU optimization patches...")
            final_path = optimize_for_npu(save_path, simplify=not args.no_simplify)
            
            if args.validate:
                source_name = "Darknet" if args.mode == 'darknet' else "PyTorch"
                yolo_status = "Raw Tensors" if (args.mode == 'darknet' and args.no_yolo_layer) else "Decoded Boxes"
                
                log_info(f"Phase 1: Comparing {source_name} and ONNX outputs ({yolo_status})...")
                is_valid = validate_conversion(model, final_path, args.shape)
                
                if args.mode == 'darknet' and not args.no_yolo_layer:
                    log_warning("YOLO decoding detected. Phase 2 (OpenCV) may fail.")
                    log_warning("If Phase 2 fails, consider using --no-yolo-layer for a cleaner export.")

                log_info(f"Phase 2: Testing inference with OpenCV DNN ({yolo_status})...")
                inference_ok = validate_with_opencv(final_path, args.shape, source_mode=args.mode)
                
                if is_valid and inference_ok:
                    log_success("All validation phases passed.")
                elif is_valid:
                    log_warning("Phase 1 passed, but Phase 2 (OpenCV) failed. This is expected for decoded YOLO layers.")
                else:
                    log_error("Validation failed.")

            print(f"\n" + "="*50)
            print(f"  CONVERSION REPORT")
            print(f"="*50)
            print(f"  Source Model : {args.weights or args.cfg}")
            print(f"  Output Path  : {final_path}")
            print(f"  Input Shape  : {args.shape}")
            
            if args.validate:
                status = "PASSED" if (is_valid and inference_ok) else ("PARTIAL (Phase 1 OK)" if is_valid else "FAILED")
                print(f"  YOLO Layer   : {yolo_status}")
                print(f"  Validation   : {status}")
            
            print("="*50 + "\n")
            
        except Exception as e:
            log_error(f"Export failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()