#!/bin/bash

# Ensure virtual environment (optional but recommended)
python3 -m venv yolofaceenv
yolofaceenv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Register a Lost Child (Image)
python main.py register child_image.jpg "John Doe" 8 "Male" "+1234567890"

# Identify Found Child (Image)
python main.py identify found_child_image.jpg

# Identify Found Child (Video)
python main.py identify found_child_video.mp4