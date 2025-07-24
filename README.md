
 🩺 Pneumonia Prediction Web App

A web-based application that predicts pneumonia from chest X-ray images using state-of-the-art deep learning models. Designed to assist healthcare professionals and patients with early and accurate detection of pneumonia.

---

🚀 Features

- 📁 Upload chest X-ray images for instant analysis
- 🤖 Multiple trained CNN models for high accuracy
- 💡 Real-time predictions with detailed output
- 🧼 Clean and responsive web interface (HTML/CSS/JS)
- 🛠️ Built using Python, Flask, and modern frontend technologies

---

## 🗂️ Folder Structure

```
├── app.py                  # Main Flask application
├── current_requirements.txt
├── models/                 # Pre-trained CNN model files
├── templates/              # HTML templates
├── static/                 # CSS, JS, and images
├── instance/               # App database (if applicable)
```

---

📦 Installation & Setup

1. Clone the repository

```
git clone https://github.com/your-username/pneumonia-prediction-app.git
cd pneumonia-prediction-app
```

2. Create virtual environment (optional but recommended)

```
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies

```
pip install -r current_requirements.txt
```

4. Run the app

```
python app.py
```

5. Access in browser

```
http://127.0.0.1:5000/
```

---

🧠 How It Works

1. Upload a chest X-ray image (JPG/PNG).
2. The image is processed through one or more CNN models (e.g., ResNet, VGG).
3. The model predicts whether pneumonia is present.
4. The result is displayed with additional analysis/visuals if available.

---

✅ Requirements

- Python 3.7+
- Flask
- TensorFlow / PyTorch
- NumPy, OpenCV, Pillow
- See `current_requirements.txt` for full list

---

📌 Notes

- Ensure the model files are placed inside the `models/` directory.
- You can swap or add models as needed — just update the loading code in `app.py`.
- This is a research/educational tool and should not replace medical diagnosis.

---

📄 License

This project is licensed under the [MIT License](LICENSE).

---

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or improve.

---

👨‍⚕️ Disclaimer

> This tool is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis or treatment.

---

✨ Acknowledgements

- Chest X-ray datasets (e.g., NIH ChestX-ray14)
- Open-source CNN architectures
- Flask and Python community
