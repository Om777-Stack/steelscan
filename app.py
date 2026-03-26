import cv2
```

And in `requirements.txt` make sure you have **only** `opencv-python-headless` and **not** `opencv-python`:
```
streamlit>=1.32.0
opencv-python-headless>=4.9.0
Pillow>=10.0.0
numpy>=1.24.0
torch==2.11.0
torchvision==0.26.0
ultralytics>=8.1.0
```

The problem is `ultralytics` is pulling in `opencv-python` as a dependency which conflicts with `opencv-python-headless`. Fix this by adding to `requirements.txt`:
```
streamlit>=1.32.0
opencv-python-headless>=4.9.0
Pillow>=10.0.0
numpy>=1.24.0
torch==2.11.0
torchvision==0.26.0
ultralytics>=8.1.0
opencv-python-headless>=4.9.0
```

Actually the real fix — add this exact line to force ultralytics to use headless:
```
streamlit>=1.32.0
Pillow>=10.0.0
numpy>=1.24.0
torch==2.11.0
torchvision==0.26.0
ultralytics>=8.1.0
opencv-python-headless>=4.9.0
