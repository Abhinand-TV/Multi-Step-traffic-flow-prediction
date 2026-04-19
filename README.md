
# 🚦 Smart Traffic Prediction AI

This project is an AI-based traffic prediction system that forecasts traffic conditions and also helps users make decisions using a simple question–answer interface.

Instead of just showing numbers, it tries to answer real questions like:

* *“Should I travel now?”*
* *“Will I get stuck?”*
* *“Is traffic getting worse?”*

---

## 📌 About the Project

The main idea of this project is to combine **traffic prediction** with a **basic AI assistant**.

* A machine learning model predicts future traffic speeds
* Based on that, the system determines traffic condition (low, moderate, high)
* Then an NLP-based assistant converts that into human-friendly answers

So instead of raw data, users get **simple and useful insights**.

---

## 🛠️ Technologies Used

* Python
* Streamlit (for UI)
* PyTorch (for model)
* NumPy, Scikit-learn
* LLM API (for NLP responses)

---

## 🧠 Features

### 1. Traffic Prediction

* Predicts upcoming traffic speed based on past data
* Uses a Transformer-based model

### 2. Traffic Summary

* Average speed
* Peak and lowest speed
* Traffic status
* Trend (improving / worsening / stable)

---

### 3. AI Assistant (NLP)

You can ask questions like:

* “Is it a good time to travel?”
* “Will I face delays?”
* “Should I wait?”

The system tries to understand the question and give a relevant answer.

---

### 4. Decision-Based Output

Instead of just showing predictions, the system gives:

* Travel suggestions
* Delay expectations
* Simple explanations

---

### 5. Stable Results

Same input (time + location) gives the same output, so the system is consistent and reliable.

---

## ⚙️ How It Works

```
User selects time + location
        ↓
Model predicts traffic speed
        ↓
System calculates status + trend
        ↓
AI assistant generates response
```

---

## 📂 Project Structure

```
traffic_flow/
│
├── app.py
├── model.py
├── dataset.py
├── train.py
├── config.py
├── llm.py
├── model.pth
```

---

## 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## 📊 Dataset

This project uses the **METR-LA dataset**, which contains traffic data collected from sensors in Los Angeles.

---

## 💡 Example Questions

* “Should I leave now?”
* “Will I get stuck in traffic?”
* “Is traffic improving?”
* “Is it crowded?”

---

## 🧪 Future Improvements

* Add real-time traffic data
* Show results on a map
* Improve NLP using a proper intent classification model
* Add voice input

---

## 👨‍💻 About

Final year project focused on combining **machine learning + NLP** to build a simple traffic assistant.

---

## 📢 Conclusion

This project shows how prediction models can be combined with NLP to make systems more practical.
Instead of just predicting data, it helps users **make decisions** based on that data.

