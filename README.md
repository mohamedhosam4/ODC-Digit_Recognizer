# ODC-Digit Recognizer 🔢

A deep learning web application for handwritten digit recognition using a Convolutional Neural Network (CNN). Built and deployed as part of the ODC AI Track.

👉 **Live App**: [Streamlit App](https://odc-digitrecognizer.streamlit.app/)  
📁 **GitHub Repository**: [ODC-Digit Recognizer](https://github.com/mohamedhosam4/ODC-Digit_Recognizer)

---

## 🧠 Project Overview

This project demonstrates a classic **digit recognition** pipeline using deep learning. The model is trained on the MNIST dataset and deployed with an interactive interface using [Streamlit](https://streamlit.io/).

Users can draw digits directly in the app and receive real-time predictions powered by a CNN model.

---

## 🔧 Features

- ✍️ Draw a digit on the canvas and get instant predictions.
- 🧠 Uses a **Convolutional Neural Network (CNN)** trained on MNIST.
- ⚡ Real-time feedback with a clean and responsive UI.
- 📦 Built with **TensorFlow** and **Streamlit**.

---

## 🚀 How to Run Locally

1. **Clone the repository**:

   ```bash
   git clone https://github.com/mohamedhosam4/ODC-Digit_Recognizer.git
   cd ODC-Digit_Recognizer
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

---

## 📁 File Structure

```
ODC-Digit_Recognizer/
├── app.py                # Streamlit app script
├── model/                # Trained model saved here
├── utils.py              # Helper functions for preprocessing
├── requirements.txt      # Project dependencies
└── README.md             # Project description and instructions
```

---

## 📚 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- OpenCV

---

## 🤝 Contributors

- **Mohamed Hosam** – [@mohamedhosam4](https://github.com/mohamedhosam4)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🌟 Acknowledgments

This project was built during the **ODC AI Training Track**. Special thanks to the instructors and peers for their support and feedback.
