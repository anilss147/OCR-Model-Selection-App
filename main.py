from flask import Flask, render_template, request, jsonify
import easyocr
import pytesseract
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize EasyOCR Reader
easyocr_reader = easyocr.Reader(['en'], gpu=False)  # Set `gpu=True` if you have a GPU and want to use it

# Configure Tesseract OCR path (update this path if Tesseract is installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Dummy CRNN model (replace with actual CRNN model if available)
class DummyCRNNModel:
    def __init__(self):
        pass

    def predict(self, image):
        return "Detected text from CRNN model"

crnn_model = DummyCRNNModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        logger.warning("No file part in the request.")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        logger.warning("No file selected for upload.")
        return jsonify({'error': 'No file selected'}), 400

    ocr_type = request.form.get('ocr_type')
    model = request.form.get('model')

    if ocr_type not in ['normal', 'deep_learning']:
        logger.warning("Invalid OCR type selected.")
        return jsonify({'error': 'Invalid OCR type selected'}), 400

    if ocr_type == 'normal' and model not in ['easyocr', 'tesseract']:
        logger.warning("Invalid model selected for Normal OCR.")
        return jsonify({'error': 'Invalid model selected for Normal OCR'}), 400

    if ocr_type == 'deep_learning' and model not in ['crnn', 'trocr']:
        logger.warning("Invalid model selected for Deep Learning OCR.")
        return jsonify({'error': 'Invalid model selected for Deep Learning OCR'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    logger.info(f"File saved to {filepath}")

    try:
        # Perform OCR based on the selected type and model
        if ocr_type == 'normal':
            if model == 'easyocr':
                logger.info("Using EasyOCR for processing.")
                result = easyocr_reader.readtext(filepath, detail=0)  # Extract text without bounding box details
                text = " ".join(result)
            elif model == 'tesseract':
                logger.info("Using Tesseract for processing.")
                text = pytesseract.image_to_string(Image.open(filepath))
        elif ocr_type == 'deep_learning':
            if model == 'crnn':
                logger.info("Using CRNN for processing.")
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((32, 100)),  # Resize to match CRNN input size
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
                text = crnn_model.predict(image_tensor)
            elif model == 'trocr':
                logger.info("Using TrOCR for processing.")
                image = Image.open(filepath).convert("RGB")
                pixel_values = processor(images=image, return_tensors="pt").pixel_values
                generated_ids = trocr_model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info("OCR processing completed successfully.")
        return jsonify({'text': text})
    except Exception as e:
        logger.error(f"Error during OCR processing: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Temporary file {filepath} removed.")

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5001)
    except Exception as e:
        logger.error(f"Failed to start the Flask application: {e}")