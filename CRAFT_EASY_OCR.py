import cv2
import easyocr
import numpy as np
from craft_text_detector import Craft


craft = Craft()


reader = easyocr.Reader(['en'])

def detect_text_craft(image_path):
    image = cv2.imread(image_path)
    prediction_result = craft.detect_text(image)
    boxes = prediction_result['boxes']
    return boxes

def recognize_text_easyocr(image_path, boxes):
    image = cv2.imread(image_path)
    results = []
    
    for box in boxes:
        
        box = np.array(box, dtype=int)
        x_min, y_min = np.min(box[:,0]), np.min(box[:,1])
        x_max, y_max = np.max(box[:,0]), np.max(box[:,1])
        
        
        cropped_image = image[y_min:y_max, x_min:x_max]
        
    
        result = reader.readtext(cropped_image)
        results.extend(result)
    
    return results

def process_image(image_path):
    boxes = detect_text_craft(image_path)
    text_results = recognize_text_easyocr(image_path, boxes)
    for result in text_results:
        print("Detected text:", result[1])

process_image('R.jpeg')
