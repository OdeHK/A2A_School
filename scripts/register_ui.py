"""
Script để chạy UI đăng ký user
Usage: python scripts/register_ui.py
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Run the UI in register mode
if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    sys.argv = ["app.py", "register"]
    
    # Import and run the app
    from ui import app
