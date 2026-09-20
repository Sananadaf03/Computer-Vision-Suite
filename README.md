# 👁️ Computer Vision Suite

## 📌 Project Overview

**Computer Vision Suite** is a multi-functional computer vision application built with **Python, OpenCV, YOLOv8, and Streamlit**.

The project combines multiple real-world computer vision applications into a single interactive platform, including **vehicle detection, parking spot analysis, trash detection, wrong-way detection, and vehicle speed estimation**.

The application provides an easy-to-use Streamlit interface where users can upload images or videos and perform different computer vision tasks.

## 🎯 Objectives

* Detect vehicles and other objects using YOLOv8.
* Analyze parking areas and parking spots.
* Detect roadside trash and litter.
* Identify vehicles moving in the wrong direction.
* Estimate vehicle speed from traffic videos.
* Provide an interactive computer vision dashboard.
* Demonstrate practical applications of AI and computer vision.

## ✨ Features

### 🚗 Vehicle Detection

Uses **YOLOv8** for object detection in images and videos.

The system can identify objects such as:

* Cars
* Motorcycles
* Buses
* Trucks
* Other supported YOLO classes

### 🅿️ Parking Spot Detection

Analyzes parking-lot images to identify parking spaces and determine their occupancy status.

This can be useful for:

* Smart parking systems
* Parking management
* Automated parking monitoring

### 🗑️ Trash Detection

Detects litter and waste in images using computer vision techniques.

Potential applications include:

* Smart city monitoring
* Road cleanliness monitoring
* Automated waste detection

### 🚦 Wrong-Way Detection

Analyzes traffic videos to identify vehicles that are moving in an incorrect direction.

This feature can be useful for:

* Traffic monitoring
* Road safety systems
* Intelligent transportation systems

### 🏎️ Speed Estimation

Processes traffic video to estimate the speed of detected vehicles.

The system uses object detection and tracking information to estimate vehicle movement over time.

## 🛠️ Technologies Used

* **Python**
* **YOLOv8**
* **Ultralytics**
* **OpenCV**
* **Streamlit**
* **NumPy**
* **Computer Vision**
* **Object Detection**
* **Object Tracking**

## 🏗️ Project Structure

```text
Computer-Vision-Suite/
│
├── app.py
├── yolov8_detector.py
├── parking_spot.py
├── trash_detection.py
├── wrong_way_detection.py
├── speed_estimation.py
├── create_test_assets.py
├── create_zip.py
│
├── test_assets/
│   ├── sample_traffic_speed.mp4
│   ├── sample_parking_lot.jpg
│   ├── sample_traffic_wrongway.mp4
│   └── sample_road_litter.jpg
│
├── static/
│   └── style.css
│
├── requirements.txt
└── README.md
```

## 🔄 Application Workflow

```text
User
  ↓
Streamlit Interface
  ↓
Select Computer Vision Task
  ↓
Upload Image / Video
  ↓
OpenCV / YOLOv8 Processing
  ↓
Object Detection / Analysis
  ↓
Display Results
```

## 🧠 YOLOv8 Object Detection

The project uses **YOLOv8** through the Ultralytics framework for real-time object detection.

General workflow:

```text
Input Image / Video
        ↓
YOLOv8 Model
        ↓
Object Detection
        ↓
Bounding Boxes
        ↓
Class Labels + Confidence
        ↓
Processed Output
```

## 📊 Modules

| Module                   | Purpose                     |
| ------------------------ | --------------------------- |
| `app.py`                 | Main Streamlit application  |
| `yolov8_detector.py`     | YOLO-based object detection |
| `parking_spot.py`        | Parking-space analysis      |
| `trash_detection.py`     | Trash/litter detection      |
| `wrong_way_detection.py` | Wrong-way vehicle detection |
| `speed_estimation.py`    | Vehicle speed estimation    |
| `create_test_assets.py`  | Creates test assets         |
| `static/style.css`       | Application styling         |

## 📁 Test Assets

Sample files are included for testing different modules:

```text
test_assets/
│
├── sample_traffic_speed.mp4
├── sample_parking_lot.jpg
├── sample_traffic_wrongway.mp4
└── sample_road_litter.jpg
```

These assets can be used to demonstrate the different features of the application.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sananadaf03/Computer-Vision-Suite.git
```

### 2. Open the Project

```bash
cd Computer-Vision-Suite
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 💻 Example Use Cases

### Smart Traffic Monitoring

The system can assist with:

* Vehicle detection
* Wrong-way vehicle detection
* Speed estimation
* Traffic video analysis

### Smart Parking

Parking-space analysis can be used to monitor:

* Available parking spaces
* Occupied spaces
* Parking-lot utilization

### Smart City Monitoring

Trash detection can help identify litter in:

* Roads
* Public areas
* Urban environments

## 🔮 Future Improvements

* Add real-time CCTV camera support.
* Improve vehicle tracking using advanced tracking algorithms.
* Add automatic number plate recognition.
* Improve speed calibration using real-world camera parameters.
* Add database storage for detection results.
* Add analytics dashboards and historical reports.
* Deploy the application as a cloud-based computer vision service.

## 📌 Project Type

**Computer Vision | Artificial Intelligence | Deep Learning | Object Detection | YOLOv8 | OpenCV**

## 👩‍💻 Author

**Sana L. Nadaf**

B.E. Computer Science & Engineering

GitHub: `Sananadaf03`
