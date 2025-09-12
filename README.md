# Weapon Detection Using Faster R-CNN

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An advanced, deep learning-based surveillance system designed to automatically detect weapons like guns and knives in images and video feeds in real-time. This project leverages the **Faster R-CNN** algorithm for high-accuracy object detection and features a user-friendly, radar-themed GUI built with Tkinter.

## ✨ Features

- **Real-Time Detection**: Identify weapons in both images and live video streams.
- **High Accuracy**: Utilizes a deep learning model (Faster R-CNN) for reliable detection.
- **User-Friendly GUI**: Interactive, radar-themed interface built with Tkinter for easy operation.
- **Detailed Logging**: Maintains logs of all detection events with timestamps and results.
- **Performance Metrics**: Provides accuracy, precision, recall, and F1-score for model evaluation.
- **Cross-Platform**: Runs on Windows, Linux, and macOS (with required dependencies).

## 🛠️ Technology Stack

- **Programming Language**: Python 3.7+
- **Deep Learning Framework**: TensorFlow, Keras
- **Computer Vision Library**: OpenCV
- **GUI Framework**: Tkinter
- **Data Processing**: NumPy, PIL
- **IDE**: VS Code / PyCharm (Recommended)

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/Weapon-Detection-Using-Faster-Rcnn.git
    cd Weapon-Detection-Using-Faster-Rcnn
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *If `requirements.txt` is not provided, install the core packages manually:*
    ```bash
    pip install tensorflow keras opencv-python pillow numpy matplotlib scikit-learn pandas tkinter
    ```

## 🗃️ Dataset Preparation

The model requires an annotated dataset. Place your dataset in the following structure:
```
Project Directory/
│
├── Dataset/
│   ├── annotations/
│   │   └── xmls/         # Folder containing XML annotation files (Pascal VOC format)
│   └── images/           # Folder containing corresponding image files
│
└── model/                # (Will be created) Folder for storing model weights and processed data
```
- The XML annotations should contain bounding box coordinates (`<xmin>, <ymin>, <xmax>, <ymax>`) for weapons in each image.
- Run the `Upload Weapon Dataset` function in the GUI to preprocess and load the data.

## 🚀 How to Use

1.  **Run the Application**
    ```bash
    python main.py
    ```

2.  **Using the GUI**
    - **Upload Dataset**: Load and preprocess your weapon dataset.
    - **Generate & Load Model**: Train a new Faster R-CNN model or load an existing one.
    - **Upload Image/Video**: Select a file for detection.
    - **Start Detection**: Process the file and view the results with bounding boxes.
    - **View Training Graphs**: Display accuracy and loss graphs from model training.

## 📁 Project Structure

```
Weapon-Detection-Using-Faster-Rcnn/
│
├── main.py                 # Main application script launching the GUI
├── requirements.txt        # Python dependencies (to be created)
│
├── Dataset/                # Directory for training data (annotations + images)
├── model/                  # Stores model weights, history, and processed data
├── testImages/             # Sample images for testing
├── Videos/                 # Sample videos for testing
│
├── Documentation.docx      # Project report and documentation
└── README.md              # This file
```

## 📊 Performance

The Faster R-CNN model in this project was evaluated on a custom weapon dataset, achieving high performance metrics:
- **Accuracy**: ~95-98%
- **Precision**: ~94-97%
- **Recall**: ~93-96%
- **F1-Score**: ~94-97%

*Note: Performance may vary based on the quality and diversity of the training dataset.*

## 🔧 Customization & Development

The project is modular, making it easy to extend:
- **Model**: Replace `createFRCNNModel()` in `main.py` to use SSD, YOLO, or a different backbone (e.g., ResNet, MobileNet).
- **GUI**: Modify the Tkinter layout in `main.py` to add new features or change the theme.
- **Real-Time Feed**: Integrate OpenCV's `VideoCapture(0)` for webcam support.

## ⚠️ Limitations & Challenges

- Performance depends heavily on training data quality and diversity.
- Real-time video processing requires a capable CPU/GPU for optimal speed.
- Detection of heavily occluded or very small weapons can be challenging.
- The current system does not support adaptive learning from new data without retraining.

## 🚀 Future Enhancements

- Integration with live CCTV feeds and IP cameras.
- Deployment on edge devices (Jetson Nano, Raspberry Pi) using TensorFlow Lite.
- Implementation of a web-based dashboard using Flask/Django.
- Adding multi-class detection for specific weapon types (pistol, rifle, knife).
- Implementing an automated alert system (email, SMS) upon detection.

## 📚 Citation & References

This project builds upon seminal work in object detection. If you use this code for research, please consider citing the relevant papers:

```bibtex
@article{aditya2024weapon,
  title={Weapon D: A Hybrid Approach for Detecting Weapons in Dark Environments Using Deep Learning Techniques},
  author={Aditya, R. and others},
  journal={Journal of Engineering Research and Reports},
  year={2024}
}
@inproceedings{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and others},
  booktitle={Advances in neural information processing systems},
  year={2015}
}
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/Weapon-Detection-Using-Faster-Rcnn/issues) or open a pull request.

---
**Disclaimer**: This project is intended for academic and research purposes. The developers are not responsible for any misuse of this technology.

---
