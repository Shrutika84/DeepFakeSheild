
## 🛡️ DeepFakeShield: AI-Powered Detection of Fake Images

![demo](https://github.com/user-attachments/assets/644d8c8e-26e6-48ea-bab7-8590324934f7)

### 🔍 About the Project

**DeepFakeShield** is a Python-based deep learning app that detects whether an uploaded image is real or AI-generated. It uses a Convolutional Neural Network (CNN) for binary classification and integrates a Large Language Model (LLM) to explain why the image was classified as real or fake — making AI decisions more transparent and interpretable.

---

### 🚀 Features

* ✅ Upload any image via a clean Streamlit UI
* 🧠 Real vs. Fake prediction using a trained CNN
* 💬 Explanation powered by an LLM (Falcon/Mistral via Hugging Face)
* 📊 Model training, testing, and evaluation scripts included
* 📂 Modular structure — ready for deployment or research extension

---

### 🧱 Project Structure

```
DeepFakeShield/
├── app/                  # Streamlit interface
│   └── app.py
├── src/                  # Core logic (models, training, inference)
│   ├── models.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── explain_llm.py
├── models/               # Trained model (.pth)
├── Dataset/              # Real & fake image folders
├── .env                  # Hugging Face token
├── requirements.txt
└── README.md
```

---

### ⚙️ Setup Instructions

#### ✅ 1. Clone the repo

```bash
git clone https://github.com/yourusername/DeepFakeShield.git
cd DeepFakeShield
```

#### ✅ 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # or venv\\Scripts\\activate on Windows
```

#### ✅ 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### ✅ 4. Add Hugging Face token

Create a `.env` file in the root:

```
HUGGINGFACEHUB_API_TOKEN= your_token_here
```

#### ✅ 5. Train the model (or skip if already trained)

```bash
python src/train.py
```

#### ✅ 6. Run the app

```bash
streamlit run app/app.py
```

---

### 🧪 Sample Test Images

Try testing with:

* Real face: from CelebA, FFHQ
* AI-generated face: from ThisPersonDoesNotExist, Midjourney

---

### 📈 Model Performance (sample)

| Model        | Accuracy | F1 Score | Notes              |
| ------------ | -------- | -------- | ------------------ |
| SimpleCNN    | 70%      | 0.68     | Small dataset demo |


---
### 🤖 LLM Explainability

Uses Hugging Face `pipeline()` with `tiiuae/falcon-rw-1b`  to explain classification in human-readable text.

---

### 🧠 Future Work

* Expand dataset with more stylized, synthetic examples
* Add Grad-CAM for visual explainability
* Extend to video-based fake detection (deepfake video)

---

### 💻 Author

**Shrutika Parab**
M.S. in Artificial Intelligence, Yeshiva University
[LinkedIn]([https://www.linkedin.com/in/your-profile](https://www.linkedin.com/in/shrutika-parab-4b6826257/))
