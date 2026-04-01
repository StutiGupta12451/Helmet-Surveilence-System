# Helmet Violation Detection System

A computer vision-based application that detects helmet violations in real-time using **YOLOv8** and provides an interactive interface via **Streamlit**.

The system identifies:

* Motorcyclists
* Riders without helmets
* Number plates (for violation tracking)

---

## Features

* Upload **images or videos**
* Detect:

  * Motorcyclists
  * No-helmet riders
  * Number plates
* Intelligent filtering:

  * Only flags **riders without helmets**
  * Links rider → number plate
* Real-time processed preview
* Download processed output (image/video)

---

## How It Works

1. The YOLO model detects objects:

   * `motorcyclist`
   * `no-helmet`
   * `plate`

2. Logic is applied:

   * Check if `no-helmet` is inside a `motorcyclist`
   * Find corresponding `plate` inside the same rider

3. If both conditions match:

   * 🚨 Mark as violation
   * Draw bounding boxes
   * Highlight number plate

---

## Tech Stack

* **Python**
* **YOLOv11 (Ultralytics)**
* **OpenCV**
* **Streamlit**

---

## Project Structure

```
helmet-detection-app/
│
├── app.py                  # Streamlit application
├── no_helmet.pt           # Trained YOLO model
├── output.jpg             # Sample output
├── requirements.txt       # Dependencies
└── README.md
```

---

## Installation

### 1️ Clone the repository

```bash
git clone https://github.com/your-username/helmet-detection-app.git
cd helmet-detection-app
```

### 2️ Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit ultralytics opencv-python numpy
```

---

## Run the App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Usage

1. Upload an image or video
2. The system processes it automatically
3. View detections
4. Download processed result

---

## Model Details

* Framework: YOLOv8 (Ultralytics)
* Custom trained classes:

  * `motorcyclist`
  * `no-helmet`
  * `plate`

---

## Limitations

* Accuracy depends on model training quality
* Plate detection may fail in:

  * Low lighting
  * Motion blur
* Does not yet include OCR (text extraction)

---

## Future Improvements

*  Add **Pytesseract OCR** for plate number extraction
* 🇮🇳 Indian number plate format validation
* Violation analytics dashboard
* Live CCTV / webcam integration
* Cloud deployment (Streamlit Cloud / HuggingFace)

---

## Contributing

Contributions are welcome!

Feel free to:

* Open issues
* Submit pull requests
* Suggest improvements

---

## License

This project is open-source and available under the MIT License.

---
