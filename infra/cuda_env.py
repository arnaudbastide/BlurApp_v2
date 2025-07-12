import os, logging, ctypes, platform

def setup_cuda():
    if platform.system() != "Windows":
        return True
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    ]
    for cp in cuda_paths:
        if os.path.isdir(cp):
            os.add_dll_directory(cp)
            logging.info("CUDA DLL dir added: %s", cp)
            return True
    logging.warning("No CUDA path found – CPU only")
    return False
setup_cuda()
