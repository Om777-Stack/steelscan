try:
    import cv2
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                   "opencv-python-headless"], check=True)
    import cv2
