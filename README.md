# 🧠 Deep Intent Mining from Student Doubt Messages

An AI-powered NLP application that analyzes student doubt messages and uncovers the underlying **topic**, **hidden intent**, **emotion**, **sentiment**, **urgency**, and **learning-risk signals**. Built with **Python**, **Streamlit**, **Scikit-learn**, and optional **Hugging Face Transformers** for advanced text understanding.

---

## 🚀 Features

### 📚 Topic Classification

Automatically identifies the academic domain of a student query:

* Mathematics
* Programming
* Data Structures
* Machine Learning
* NLP
* Database / SQL
* Computer Networks
* Operating Systems
* Electronics
* Other

### 🎯 Hidden Intent Detection

Detects the student's underlying objective:

* Concept Misunderstanding
* Confusion
* Frustration
* Exam Anxiety
* Assignment Stress
* Debugging Issues
* Need for Explanation
* Resource Request
* Deadline Pressure
* Clarification Request

### 😊 Emotion Analysis

Recognizes emotional states behind messages:

* Confusion
* Anxiety
* Frustration
* Curiosity
* Confidence
* Stress
* Sadness
* Neutral

### 💬 Sentiment Analysis

Classifies overall sentiment as:

* Positive
* Negative
* Neutral

### ⚠️ Learning Risk Assessment

Identifies students who may require immediate support based on:

* Emotional intensity
* Exam pressure
* Deadline urgency
* Frustration indicators

### 🔍 Aspect Detection

Extracts educational concerns related to:

* Concepts
* Implementation
* Exams
* Deadlines
* Resources
* Mathematics

### 📊 Batch Analytics

Analyze multiple messages simultaneously through:

* Text input
* CSV upload
* Downloadable enriched CSV results

---

## 🏗️ System Architecture

```text
User Input
    │
    ▼
Preprocessing
    │
    ▼
Semantic Engine (TF-IDF + Cosine Similarity)
    ├── Topic Detection
    └── Intent Detection
    │
    ▼
Emotion Detection
    │
    ▼
Sentiment Analysis
    │
    ▼
Risk & Aspect Analysis
    │
    ▼
Decision Fusion
    │
    ▼
Summary + Analytics Dashboard
```

---

## 🛠️ Tech Stack

| Component           | Technology                 |
| ------------------- | -------------------------- |
| Frontend            | Streamlit                  |
| NLP                 | Scikit-learn               |
| Semantic Similarity | TF-IDF + Cosine Similarity |
| Emotion Detection   | Hugging Face Transformers  |
| Sentiment Analysis  | Hugging Face Transformers  |
| Data Processing     | Pandas, NumPy              |
| Visualization       | Plotly                     |
| Language            | Python                     |

---

## 📂 Project Structure

```text
Deep-Intent-Mining/
│
├── app.py
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
│
├── sample_datasets/
│   ├── student_doubt_test_data.csv
│
└── screenshots/
    └── architecture.png
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/deep-intent-mining.git
cd deep-intent-mining
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📥 Input Methods

### 1. Single Message Analysis

Example:

```text
I am confused about backpropagation. The formulas seem easy but I keep getting wrong answers in the exam.
```

### 2. Multiple Message Analysis

Paste multiple student messages line-by-line.

### 3. CSV Upload

Upload a CSV file containing a column named:

```text
text
```

Example:

| text                          |
| ----------------------------- |
| I am confused about SQL joins |
| My code is not working        |
| I am worried about the exam   |

---

## 📈 Example Output

```text
Topic: Machine Learning

Hidden Intent:
Concept Misunderstanding

Emotion:
Anxiety

Sentiment:
Negative

Learning Risk:
High

Aspect Signals:
exam, concept, formulas
```

---

## 🎓 Educational Use Cases

* Learning Management Systems (LMS)
* Student Support Platforms
* AI Teaching Assistants
* Educational Chatbots
* Academic Risk Monitoring
* Personalized Learning Systems
* Early Detection of Student Anxiety

---

## 🔮 Future Improvements

* Deep Learning Topic Classification
* Multi-Language Support
* Instructor Alert System
* Student Recommendation Engine
* Real-Time API Deployment
* LLM-Powered Explanations
* Knowledge Graph Integration
* Dashboard Authentication

---

## 🤝 Contributing

Contributions are welcome!

You can help by:

* Improving NLP models
* Adding new intent categories
* Expanding emotion detection
* Supporting additional educational domains
* Enhancing UI/UX
* Creating benchmark datasets

Please read:

```text
CONTRIBUTING.md
```

before submitting pull requests.

---

## 📜 License

This project is licensed under the MIT License.

See:

```text
LICENSE
```

for details.

---

## 🌟 Community

If you find this project useful:

⭐ Star the repository

🐛 Report bugs through Issues

💡 Suggest new features

🔀 Submit Pull Requests

💬 Join Discussions

---

## 👨‍💻 Author

Developed as an Educational NLP and Learning Analytics project focused on understanding student learning behavior through intent, emotion, and risk-aware text mining.

---

