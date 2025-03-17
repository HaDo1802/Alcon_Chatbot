import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_config as cfg

# 🔎 Debugging: Check what is inside `config.py`
print("Config module attributes:", dir(cfg))
print("Config file location:", cfg.__file__)

# 🔎 Check if 'deploy' exists
if hasattr(cfg, "deploy"):
    print(f"✅ deploy exists in config.py: {cfg.deploy}")
else:
    print("❌ deploy is MISSING in config.py!")
