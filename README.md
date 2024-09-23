# YOLO: Detecting Signage and Speech Generation for the Visually Impaired

This repository contains the implementation of a deep learning-based system that leverages YOLO for detecting indoor signage and integrates text-to-speech (TTS) technology to assist visually impaired individuals by providing real-time auditory feedback about detected signs.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project introduces a system that aids visually impaired individuals in indoor navigation by recognizing critical signage, such as restrooms, exits, and elevators, and converting the detected information into speech. The system employs the YOLO object detection algorithm combined with TTS technologies like gTTS or pyttsx3 to generate real-time auditory descriptions of signage.

This tool enhances the independence and safety of visually impaired individuals by allowing them to navigate complex environments without assistance.

## Features

- **Real-time object detection** using YOLO.
- **Signage detection** in images or video feeds.
- **Text-to-speech (TTS)** generation for accessibility.
- Easy to extend and modify for other object detection tasks.
  
## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/T-Chaitanya/YOLO-Detecting-Signage-and-Speech-Generation.git
    cd YOLO-Detecting-Signage-and-Speech-Generation
    ```

2. **Install the dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download YOLO pre-trained weights**

    Download the pre-trained YOLO weights file from the [official YOLO site](https://pjreddie.com/darknet/yolo/), and place it in the `weights/` directory.

## Usage

1. **Signage Detection**
    - To detect signage in an image:
      ```bash
      python detect.py --image_path <path_to_image>
      ```

    - To detect signage in a video:
      ```bash
      python detect.py --video_path <path_to_video>
      ```

2. **Speech Generation**
    - After detecting signage, the recognized text will be converted to speech automatically using the `gTTS` (Google Text-to-Speech) library.
  
3. **Output**
    - The detected signage will be highlighted in the output image/video and the generated speech will be played.

## Model Training

If you wish to train the YOLO model with a custom dataset:

1. **Prepare your dataset** following YOLO format.
2. **Modify the configuration** for YOLO in the `cfg/` directory to match your dataset.
3. **Train the model** by running:
    ```bash
    python train.py --data <path_to_data> --config <path_to_config> --weights <path_to_pretrained_weights>
    ```

## Results

Examples of detected:
![val_batch1_labels](https://github.com/user-attachments/assets/09ede7d5-c7a7-48c8-bf3a-b6ddaf7597ae)

## Dependencies

- Python 3.7+
- OpenCV
- TensorFlow / PyTorch (for YOLO implementation)
- `gTTS` (Google Text-to-Speech)
- Other dependencies listed in `requirements.txt`

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
