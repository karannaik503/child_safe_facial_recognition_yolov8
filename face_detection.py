import cv2
import numpy as np
import os
import traceback
import logging
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path=None):
        """
        Initialize face detector with comprehensive logging
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.weights_dir = "weights"
        
        # Determine model path
        if model_path is None:
            model_path = os.path.join(self.weights_dir, "yolov8n-face.pt")
        
        # Validate model file exists
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}. Please download the YOLOv8 face detection model.")
        
        try:
            self.model = YOLO(model_path)
            self.logger.info("Face detection model loaded successfully!")
        except Exception as e:
            self.logger.error(f"Error loading face detection model: {e}")
            raise

    def detect_faces_in_image(self, image_path_or_array):
        """
        Detect faces with comprehensive diagnostics
        
        Args:
            image_path_or_array (str or numpy.ndarray): Image source
        
        Returns:
            list: Detected face images
        """
        try:
            # Handle both file path and numpy array input
            if isinstance(image_path_or_array, str):
                image = cv2.imread(image_path_or_array)
                if image is None:
                    self.logger.error(f"Could not read image at {image_path_or_array}")
                    return []
            else:
                image = image_path_or_array

            # Diagnostic image information
            self.logger.info(f"Image shape: {image.shape}")
            self.logger.info(f"Image dtype: {image.dtype}")

            # Run detection
            results = self.model(image)
            
            faces = []
            for r in results:
                boxes = r.boxes
                self.logger.info(f"Number of detected boxes: {len(boxes)}")
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Additional validation
                    if x2 <= x1 or y2 <= y1:
                        self.logger.warning("Invalid bounding box coordinates")
                        continue
                    
                    face = image[y1:y2, x1:x2]
                    
                    if face.size > 0:
                        # Resize face consistently
                        face = cv2.resize(face, (160, 160))
                        faces.append(face)
                        
                        # Log face extraction details
                        self.logger.info(f"Extracted face: {face.shape}")
            
            self.logger.info(f"Total faces detected: {len(faces)}")
            return faces
        
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            self.logger.error(traceback.format_exc())
            return []

def detect_faces(input_path, is_video=False, output_path=None):
    """
    Unified face detection function with enhanced logging
    """
    detector = FaceDetector()
    
    if is_video:
        # Video processing
        faces = []
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logging.error(f"Could not open video file: {input_path}")
            return []
        
        # Process multiple frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        logging.info(f"Video details - Frames: {frame_count}, FPS: {fps}, Duration: {duration} seconds")
        
        # Sample frames (every second)
        sample_interval = max(1, int(fps))
        
        for frame_idx in range(0, frame_count, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_faces = detector.detect_faces_in_image(frame)
            faces.extend(frame_faces)
        
        cap.release()
        return faces
    else:
        # Image processing
        return detector.detect_faces_in_image(input_path)