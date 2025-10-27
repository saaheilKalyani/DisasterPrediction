# Multi-Disaster Prediction System

## Project Definition
This project is an AI-based disaster prediction system that forecasts the likelihood of three natural disasters — Flood, Cyclone, and Earthquake.  
It uses trained machine learning models to analyze environmental parameters and predict the probability and risk level of each disaster.  
The project includes automated data preprocessing, model training, evaluation, and a user interface built with Streamlit for easy interaction.

---

## Features
- Predicts three disasters: Flood, Cyclone, and Earthquake  
- Machine learning models trained using Random Forest and SVM  
- Automated preprocessing and model loading  
- Streamlit-based graphical user interface  
- Likelihood score and risk level prediction  
- Supports both CSV file upload and manual input  

---

## Project Structure
```
DisasterPrediction/
├─ app.py                     # Streamlit user interface
├─ src/
│  ├─ preprocess.py           # Preprocessing and feature generation
│  ├─ train_model.py          # Model training and evaluation
│  ├─ predict.py              # Command-line prediction script
├─ models/
│  ├─ flood/
│  ├─ cyclone/
│  └─ earthquake/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ requirements.txt
└─ README.md
```

---

## Requirements
- Python 3.9 or higher  
- pip package manager  
- Internet connection for installing dependencies  

---

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/DisasterPrediction.git
cd DisasterPrediction
```

### Step 2: Create and Activate a Virtual Environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

Once started, Streamlit will open the app in your browser (default: http://localhost:8501)

---

## Usage Instructions
1. Select the disaster type (Flood, Cyclone, or Earthquake) from the dropdown menu.  
2. Enter feature values manually or upload a CSV file containing input data.  
3. Click on the **Predict** button.  
4. The system will display:
   - Predicted Likelihood Score (0–100)
   - Risk Level (Low, Medium, or High)

---

## Testing

### Run Local Prediction Test
Use the command-line prediction script to test model inference.

```bash
python src/predict.py --disaster flood --input data/processed/flood/test.csv --output results.csv
```

**Expected output:**
- No errors or warnings  
- File `results.csv` created with predicted scores and risk levels  

### UI Verification
```bash
streamlit run app.py
```

Then:  
1. Open the browser link provided  
2. Test all three disaster options  
3. Verify that predictions display correctly for each disaster  

---

## Deployment

### Option 1: Deploy on Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)  
2. Choose **Streamlit** as the runtime  
3. Upload this repository or connect your GitHub repository  
4. Ensure `app.py` and `requirements.txt` are in the root directory  
5. The app will automatically build and go live  

### Option 2: Deploy on Streamlit Cloud
1. Push the repository to GitHub  
2. Go to [Streamlit Cloud](https://share.streamlit.io)  
3. Create a new app and select your repository  
4. Set `app.py` as the entry point  
5. Deploy and open the provided URL  

---

## Verification After Deployment
- Application runs without errors  
- Model predictions consistent with local results  
- Public URL accessible  
- Each disaster model returns a valid prediction  

---

## Author
**Name:** Saaheil Kalyani  
**Role:** Developer & Researcher – AI and Machine Learning  
**Project:** Multi-Disaster Prediction System  
**Description:** Designed and developed an AI-based predictive system capable of analyzing environmental data to forecast multiple natural disasters (Flood, Cyclone, Earthquake) using machine learning models and a Streamlit interface for public deployment.
