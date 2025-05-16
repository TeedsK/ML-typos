import torch

print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    # Check CUDA version PyTorch was compiled with
    print(f"PyTorch built with CUDA version: {torch.version.cuda}")
    # You can also try a small tensor operation on CUDA
    try:
        tensor = torch.tensor([1.0, 2.0]).cuda()
        print(f"Test tensor on CUDA: {tensor}")
        print("CUDA is working with PyTorch!")
    except Exception as e:
        print(f"Error during CUDA test operation: {e}")
else:
    print("CUDA is not available. PyTorch will use CPU.")
    print("Ensure you have the correct PyTorch version installed for your CUDA toolkit version.")
    print("You might need to reinstall PyTorch with CUDA support if it's not detected.")
    print("Installation command usually looks like: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX")
    print("(Replace cuXXX with your CUDA version, e.g., cu118 for CUDA 11.8, or cu121 for CUDA 12.1)")