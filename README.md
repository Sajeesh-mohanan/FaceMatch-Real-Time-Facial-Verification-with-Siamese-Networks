# Siamese Neural Network for Face Verification 🔍🧠

This project implements a **Siamese Neural Network** using TensorFlow for real-time **face verification**. The goal is to determine whether two facial images belong to the same person.

![Research-paper][text](oneshot1.pdf)


## 📌 Objective

Build and train a deep learning model capable of learning face embeddings and verifying identity based on facial similarity. This has real-world applications in:

- 🔐 Authentication systems  
- 🧑‍💻 Access control  
- 🧠 Few-shot learning setups


## 📌 Project Overview

- Capture face images from a webcam for Anchor and Positive samples.
- Use the **Labeled Faces in the Wild (LFW)** dataset as Negative samples.
- Preprocess and format the dataset for training.
- Build a **custom Siamese model** using TensorFlow's Functional API.
- Train the model to learn **face embeddings** and compute similarity.
- Perform real-time face verification using the trained model.


## 🗂️ Project Structure

├── data/
│ ├── anchor/ # Custom face images (reference)
│ ├── positive/ # Matching face images
│ ├── negative/ # Non-matching images from LFW
│
├── lfw.tgz # LFW dataset (tar file)
├── siamesemodelv2.h5 # (Optional) Trained model due to
├── Facial_verification_using_Siamese_Network.ipynb # Main Jupyter Notebook
└── README.md # Project description

## 📊 Dependencies

- Python ≥ 3.7  
- TensorFlow  
- NumPy  
- OpenCV  
- Matplotlib  
- uuid, os, shutil

## 🧪 Features & Workflow

1. **Data Collection**
   - Webcam captures anchor & positive face images.
   - LFW dataset used for negative examples.

2. **Preprocessing**
   - Images resized and normalized.
   - Triplet dataset generated (Anchor, Positive, Negative).

3. **Model Architecture**
   - Built using **TensorFlow Functional API**.
   - Custom embedding model with shared weights.
   - Distance layer for similarity comparison.

4. **Training**
   - Loss: **Contrastive loss** or **Triplet Loss**.
   - Training loop uses TensorFlow’s `tf.GradientTape`.

5. **Verification**
   - Once trained, the model verifies identity in real-time.
   - Face match threshold customizable for accuracy vs. flexibility.


## 🎮 Webcam Controls

| Key | Action                  |
|-----|--------------------------|
| `A` | Capture **Anchor** image |
| `P` | Capture **Positive** image |
| `Q` | Quit webcam feed         |

---

## 📈 Example Output

After training, the model can verify if a test image matches a known person based on learned facial embeddings.

## 🧠 Future Improvements

- Add GUI interface for user-friendly verification.
- Expand dataset for multi-person classification.
- Implement model saving and reloading pipeline.

---

## 🙌 Author

**Sajeesh K Mohanan** 
🔗 [LinkedIn](https://www.linkedin.com/in/sajeesh-k-mohanan-7a437218b/)  