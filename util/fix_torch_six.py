"""
timm 0.3.2와 최신 PyTorch 호환성 패치
PyTorch 1.9+ 에서 제거된 torch._six 모듈을 패치합니다.

사용법: 다른 모듈을 import하기 전에 이 파일을 먼저 import하세요.
    import util.fix_torch_six
"""

import collections.abc

# torch._six가 없는 경우를 위한 패치
try:
    import torch
    if not hasattr(torch, '_six'):
        class _Six:
            container_abcs = collections.abc
        torch._six = _Six()
except ImportError:
    # torch가 없는 경우는 패스
    pass

