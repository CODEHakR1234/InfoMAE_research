#!/usr/bin/env python
"""
timm 0.3.2와 최신 PyTorch 호환성 패치
torch._six 문제를 해결합니다.
"""

import sys
import collections.abc

# torch._six가 없는 경우를 위한 패치 적용
try:
    import torch
    if not hasattr(torch, '_six'):
        print("torch._six 모듈이 없습니다. 호환성 패치 적용 중...")
        torch._six = type('Module', (), {})()
        torch._six.container_abcs = collections.abc
        print("✓ 패치 적용 완료")
    else:
        print("torch._six 모듈이 이미 존재합니다.")
except Exception as e:
    print(f"패치 적용 중 오류: {e}")

