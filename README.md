## Title: DeepActionsExperimental - ROAD OBJECT DETECTION

⬇️ **Download**

| Distribution | Format | Source |
| ------------ | ------ | ------ |
| Ubuntu,kali,Debian,Udroid       | Deb   | [Download Deep Action Experimental v1.0 alpha](https://github.com/Exploit0xfffff/DeepActionsExperimental/releases/download/v1.0/Deep-Actions-Experimental.deb) |
| Developer    | clone  | `git clone https://github.com/Exploit0xfffff/PenetrationApp` |

### 👨🏿‍🔧👨🏿‍🔧 Ubuntu Installation 👨🏿‍🔧👨🏿‍🔧

```bash
sudo dpkg -i Deep-Actions-Experimental.deb
sudo apt --fix-broken install
```

### 🤽🏾🤽🏾 Uninstallation 🤽🏾🤽🏾

```bash
sudo apt remove deep-actions-experimental
```

🔨🔨 **Developers**🔨🔨

```bash
git clone https://github.com/Exploit0xfffff/Deep-Actions-Experimental
cd Deep-Actions-Experimental
pip install -r requirements.txt
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0
cd main/
python3 main.py
```

### 🎯 Features and Models

The system now supports two powerful object detection models with enhanced performance:

- **Faster R-CNN**: Traditional model with ResNet50 backbone
  - Average FPS: 0.40 on CPU
  - Best for high-accuracy detection
  - Optimized for detailed analysis
- **YOLO**: Real-time object detection
  - Average FPS: 5.13 on CPU
  - Optimized for speed and real-time processing
  - Efficient resource utilization

### 🆕 Latest Updates
- Added YOLO model integration
- Implemented model switching capability
- Enhanced CPU optimization
- Improved hardware detection
- Added comprehensive documentation
- Updated GTK UI with model selection
- Added performance monitoring

### 💻 Hardware Support
- Automatic hardware detection and optimization
- CPU-specific performance enhancements
- Real-time FPS counter and metrics
- Dynamic resource allocation
- Optimized batch processing

### 📊 Performance Metrics
- **YOLO Performance**:
  - CPU: 5.13 FPS average
  - Inference time: 0.195s
- **Faster R-CNN Performance**:
  - CPU: 0.40 FPS average
  - Inference time: 2.527s

ui  ->

![image](https://github.com/Exploit0xfffff/DeepActionsExperimental/assets/81065703/7fab3f6d-5603-40e9-957c-244236b49c96)

Image Capture before ->

![OIP (1)](https://github.com/Exploit0xfffff/Deep-Actions-Experimental/assets/81065703/76b033a5-4882-44c7-9924-6a5d2faa7095)

Image Capture after ->

![OIP (1)_output](https://github.com/Exploit0xfffff/Deep-Actions-Experimental/assets/81065703/41375075-2686-4f5a-9ae6-930d2c8d36b9)

Video Capture before ->

https://github.com/Exploit0xfffff/Deep-Actions-Experimental/assets/81065703/d8a39a03-2458-4f28-9099-15a227669734

Video Capture after ->

https://drive.google.com/file/d/181Gdxq9ruSSTRAvR18yt7yFdWgrPmN_9/view?usp=sharing

live capture ->

![Screenshot from 2023-06-12 21-20-33](https://github.com/Exploit0xfffff/DeepActions-Experimental/assets/81065703/66d208f3-1ce2-4f82-8a2b-19b021f2c57b)

### 🚀 Development Roadmap
Current improvements:
- Enhanced object detection pipeline
- Multi-model support (YOLO + Faster R-CNN)
- Improved CPU optimization
- Updated GTK UI with model selection
- Comprehensive documentation

Future plans:
- Integration of Google Maps and Google Lens
- Mobile application development
- Further YOLO optimization
- Enhanced real-time capabilities
- Advanced hardware optimization

Please note that this project now includes both traditional and modern AI approaches, combining GTK (Graphical User Interface Toolkit), OpenCV (Open Source Computer Vision Library), the COCO (Common Objects in Context) dataset, and multiple detection models (R-CNN and YOLO) connected to both CPU and GPU. The UI has been updated with model selection capabilities, and the latest modules have been incorporated into the project.

### 🚀 Performance Notes
- Faster R-CNN: Better for detailed analysis and high accuracy requirements
- YOLO: Optimized for real-time detection and faster processing
- Both models utilize the COCO dataset for consistent object detection capabilities

### 🔧 System Requirements
- Python 3.10 or higher
- GTK 3.0
- OpenCV
- PyTorch 2.2.0+
- CUDA (optional for GPU support)

### 📚 Documentation
The codebase now includes comprehensive comments and documentation:
- Detailed implementation explanations
- Step-by-step guides
- Performance optimization tips
- Hardware configuration guides

Feel free to explore the project and provide any feedback or suggestions. Contributions are always welcome!
