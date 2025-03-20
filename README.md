# OCR Application with Model Selection

This is a Flask-based OCR (Optical Character Recognition) application that allows users to upload images and extract text using different OCR models. The application supports both normal OCR models (e.g., EasyOCR, Tesseract) and deep learning-based OCR models (e.g., CRNN, TrOCR). Future models like YOLO and BART are also listed but not yet implemented.

## Features
- **Normal OCR Models**:
  - EasyOCR
  - Tesseract
- **Deep Learning Models**:
  - CRNN
  - TrOCR
- **Future Models**:
  - YOLOv3, YOLOv4, YOLOv5
  - BART (Future)

## Prerequisites
Before running the application, ensure the following are installed on your system:
1. **Python**: Version 3.8 or higher.
2. **Pip**: Python's package manager.
3. **Tesseract OCR**: Required for the Tesseract model.

### Installing Tesseract OCR
- Download and install Tesseract OCR from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract).
- After installation, note the installation path (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe` on Windows).
- Update the `pytesseract.pytesseract.tesseract_cmd` path in `main.py` to point to your Tesseract installation.

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
