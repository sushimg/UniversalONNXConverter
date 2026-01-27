import argparse
import os
import sys
import torch

from core.darknet_parser import DarknetParser
from exporters.engine import export_pytorch_to_onnx, try_export_ultralytics
from exporters.post_process import optimize_for_npu
from validators.compare import validate_conversion

def get_args():
    parser = argparse.ArgumentParser(
        description="Universal NPU Model Converter: Convert Darknet or PyTorch models to NPU-optimized ONNX format.",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )
    
    group = parser.add_argument_group('Required Arguments')
    group.add_argument('--mode', type=str, choices=['darknet', 'pytorch'], 
                        help='Input model type:\n'
                             '  darknet: Use .cfg and .weights files\n'
                             '  pytorch: Use .pt files')
    group.add_argument('--weights', type=str, help='Path to weight file (.weights or .pt)')
    
    group = parser.add_argument_group('Optional Settings')
    group.add_argument('--cfg', type=str, help='Path to Darknet .cfg file (Required ONLY for darknet mode)')
    group.add_argument('--output', type=str, default='model.onnx', help='Output file name/path (Default: model.onnx)')
    group.add_argument('--output_mode', type=str, default='onnx', choices=['onnx', 'pt'],
                        help='Output format:\n'
                             '  onnx: Optimized for NPU (Default)\n'
                             '  pt  : Standard PyTorch save')
    group.add_argument('--shape', type=int, nargs=4, default=[1, 3, 416, 416], 
                        help='Input Shape: Batch Channel Height Width\nExample: 1 3 640 640 (Default: 1 3 416 416)')
    group.add_argument('--opset', type=int, default=18, help='ONNX Opset version (11 or 18 recommended for NPU)')
    
    group = parser.add_argument_group('ONNX Specific Flags')
    group.add_argument('--validate', action='store_true', help='Compare PyTorch vs ONNX outputs for numerical correctness')
    group.add_argument('--no-simplify', action='store_true', help='Skip the onnx-simplifier step')

    group = parser.add_argument_group('Information')
    group.add_argument('--tutorial', action='store_true', help='Show example commands for all conversion scenarios')
    group.add_argument('--help', '-h', action='help', help='Show this help message and exit')

    args = parser.parse_args()
    return args

def show_tutorial():
    tutorial_path = os.path.join(os.path.dirname(__file__), 'commands', 'tutorial.txt')
    if os.path.exists(tutorial_path):
        print("\n" + "="*50)
        print("          NPU CONVERTER USE-READY COMMANDS")
        print("="*50)
        with open(tutorial_path, 'r') as f:
            print(f.read())
        print("="*50 + "\n")
    else:
        print("[ERROR] Tutorial file not found at commands/tutorial.txt")

def load_model(args):
    print(f"Loading model... Mode: {args.mode.upper()}")
    
    model = None
    
    if args.mode == 'darknet':
        if not args.cfg:
            print("[ERROR] Darknet mode requires --cfg file!")
            sys.exit(1)
        
        try:
            model = DarknetParser(args.cfg)
            model.load_weights(args.weights)
            print(f"[SUCCESS] Darknet model initialized.")
        except Exception as e:
            print(f"[ERROR] Failed to load Darknet model: {e}")
            sys.exit(1)

    elif args.mode == 'pytorch':
        print(f"Reading PyTorch file: {args.weights}")
        try:
            try:
                loaded = torch.load(args.weights, map_location='cpu', weights_only=False)
            except TypeError:
                loaded = torch.load(args.weights, map_location='cpu')

            if isinstance(loaded, dict):
                if 'model' in loaded:
                    model = loaded['model']
                    print("[SUCCESS] Checkpoint structure detected (likely Ultralytics).")
                    
                    if hasattr(model, 'float'):
                        model.float()
                elif 'state_dict' in loaded:
                    print("[ERROR] This file only contains weights (state_dict), architecture is missing.")
                    sys.exit(1)
                else:
                    model = loaded
            else:
                model = loaded
            
            if hasattr(model, 'fuse'):
                try:
                    print("Applying model 'fuse' operation...")
                    model.fuse()
                except:
                    pass
                
            print("[SUCCESS] PyTorch model loaded into memory.")
            
        except ModuleNotFoundError as e:
            print(f"[ERROR] Missing library: {e}. (pip install ultralytics might be required)")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] File could not be loaded: {e}")
            sys.exit(1)

    if hasattr(model, 'eval'):
        model.eval()
        
    return model

def main():
    args = get_args()

    if args.tutorial:
        show_tutorial()
        return

    if not args.mode or not args.weights:
        print("[ERROR] Arguments --mode and --weights are required unless using --tutorial.")
        print("Use --help for more information.")
        return

    if args.output_mode == 'pt':
        if args.validate:
            print("[WARNING] --validate is ignored in PyTorch output mode.")
        if args.no_simplify:
            print("[WARNING] --no-simplify is ignored in PyTorch output mode.")
    
    model = load_model(args)

    if args.output_mode == 'pt':
        print(f"Process: Converting to PyTorch format (.pt)")
        save_path = args.output if args.output.endswith('.pt') else args.output + '.pt'
        try:
            torch.save(model, save_path)
            print(f"[SUCCESS] Model saved -> {save_path}")
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")

    elif args.output_mode == 'onnx':
        print(f"Process: ONNX Export (NPU Optimized)")
        save_path = args.output if args.output.endswith('.onnx') else args.output + '.onnx'

        try:
            is_ultralytics = try_export_ultralytics(
                model=model,
                input_shape=args.shape,
                output_path=save_path,
                opset=args.opset
            )
            
            if not is_ultralytics:
                print("Standard model structure detected, using generic export.")
                export_pytorch_to_onnx(
                    model=model,
                    input_shape=args.shape,
                    output_path=save_path,
                    opset=args.opset
                )
            
            print("Applying NPU compatibility patches...")
            final_path = optimize_for_npu(
                onnx_path=save_path,
                target_ir=8,
                simplify=not args.no_simplify
            )
            
            if args.validate:
                print("Performing validation...")
                is_valid = validate_conversion(
                    pt_model=model,
                    onnx_path=final_path,
                    input_shape=args.shape
                )
                if is_valid:
                    print(f"\n[SUCCESS] Model is fully ready for NPU!")
                else:
                    print(f"\n[WARNING] Validation differences occurred (may be normal due to head change).")
            else:
                print(f"\n[SUCCESS] Process completed -> {final_path}")
                
        except Exception as e:
            print(f"\n[ERROR] EXPORT FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

