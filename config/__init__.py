# config/__init__.py
"""
Package init file cho config module.
Cho phép import trực tiếp các settings và constants từ package config.
"""

from .settings import Settings, get_settings
from .constants import ModelConstants, UIConstants, FileConstants

# Export các class và function chính để dễ import
__all__ = [
    'Settings',
    'get_settings',
    'ModelConstants', 
    'UIConstants',
    'FileConstants'
]

# Tạo instance global settings để sử dụng trong toàn bộ app
settings = get_settings()
