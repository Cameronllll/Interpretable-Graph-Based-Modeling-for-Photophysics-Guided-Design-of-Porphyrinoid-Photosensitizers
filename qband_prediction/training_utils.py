"""通用的确定性训练、数据加载与模型哈希工具。"""
import hashlib
import os
import platform
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning
import torch
import torch_geometric
import torch_geometric.typing
from rdkit import rdBase
from torch.utils.data import DataLoader

from data import collate_fn


def windows_high_qos():
    """提高当前训练进程的运行服务等级，不修改系统电源设置。"""
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [('Version', wintypes.DWORD), ('ControlMask', wintypes.DWORD), ('StateMask', wintypes.DWORD)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    state = State(1, 1, 0)
    return bool(kernel.SetProcessInformation(kernel.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)))


def configure_runtime():
    """固定运算精度及确定性设置，并记录当前运行环境。"""
    high_qos = windows_high_qos()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch_geometric.typing.WITH_TORCH_SCATTER = False
    warnings.filterwarnings('ignore', message="The usage of `scatter.*torch-scatter.*")
    return {
        'deterministic_algorithms': True, 'warn_only': False,
        'matmul_precision': 'highest', 'tf32': False, 'cpu_threads': 4,
        'cudnn_benchmark': False, 'pyg_pooling_backend': 'native torch',
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'lightning': pytorch_lightning.__version__, 'pyg': torch_geometric.__version__,
        'rdkit': rdBase.rdkitVersion, 'windows_high_qos': high_qos,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'numpy': np.__version__, 'pandas': pd.__version__,
    }


def loader(dataset, batch_size, seed, shuffle=False):
    """为每个 DataLoader 使用独立随机数，避免影响模型初始化或 Dropout。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=torch.cuda.is_available())


def state_hash(state):
    """根据张量内容计算模型状态哈希，不受 checkpoint 文件元数据影响。"""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8'))
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()
