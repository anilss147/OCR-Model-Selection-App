# Setup Guide for OCR Application with Model Selection

This guide provides step-by-step instructions to set up and run the OCR application. Follow these steps to ensure everything works smoothly.

---

## **1. Prerequisites**

Before starting, ensure the following are installed on your system:

1. **Python**: Version 3.8 or higher.
   - Download Python from [python.org](https://www.python.org/downloads/).
   - Verify installation:
     ```bash
     python --version
     ```

2. **Pip**: Python's package manager (comes with Python installation).
   - Verify installation:
     ```bash
     pip --version
     ```

3. **Tesseract OCR**: Required for the Tesseract model.
   - Download and install Tesseract OCR from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract).
   - After installation, note the installation path (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe` on Windows).
   - Verify installation:
     ```bash
     tesseract --version
     ```

---

## **2. Clone the Repository**

1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   cd <repository-folder>