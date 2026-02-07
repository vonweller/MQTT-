"""
PyInstaller runtime hook for PyTorch
解决打包后 _C 未定义的问题
"""

import sys
import os

# 确保PyTorch能找到其C扩展
if hasattr(sys, '_MEIPASS'):
    # 运行在PyInstaller打包环境中
    torch_lib_path = os.path.join(sys._MEIPASS, 'torch', 'lib')
    if os.path.exists(torch_lib_path):
        # 添加torch/lib到PATH，以便加载DLL
        os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
        
    # 设置TORCH_HOME环境变量
    os.environ['TORCH_HOME'] = os.path.join(sys._MEIPASS, 'torch')
    
    # 禁用CUDA缓存（如果使用CPU版本）
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
