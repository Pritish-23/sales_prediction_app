\# 📊 Product Sales Prediction System



A machine learning–powered Streamlit application that analyzes customer reviews to predict a product’s sales performance and provide actionable business insights.



---



\## 🚀 Project Overview



This project leverages \*\*Natural Language Processing (NLP)\*\* and \*\*machine learning models\*\* to analyze product reviews and estimate sales strength.



It helps answer questions like:

\- Is this product likely to be a \*\*high-sales\*\* or \*\*low-sales\*\* product?

\- How strong is its \*\*sales performance relative to others\*\*?

\- What do customers \*\*like or dislike\*\* the most?

\- How has review activity changed over time?



The system is built as an \*\*interactive dashboard\*\* using Streamlit.



---



\## 🧠 Key Features



\- 📂 Upload CSV file containing reviews of a single product

\- 😊 Sentiment analysis using VADER

\- 📈 Review activity trends over time

\- 🟢 / 🔴 Sales category prediction (High vs Low)

\- 🔢 Sales Strength Index (normalized score)

\- 🧠 Business interpretation \& actionable insights

\- 🗣️ Highlighted positive and negative reviews

\- 📌 Key themes detection from customer feedback



---



\## 🛠️ Tech Stack



\- \*\*Language:\*\* Python  

\- \*\*Frontend:\*\* Streamlit  

\- \*\*Data Processing:\*\* Pandas, NumPy  

\- \*\*NLP:\*\* NLTK (VADER Sentiment Analyzer)  

\- \*\*Machine Learning:\*\* Scikit-learn, XGBoost  

\- \*\*Model Persistence:\*\* Joblib  



---



\## 📂 Project Structure


sales\_prediction\_app/

│

├── app.py

├── requirements.txt

├── README.md

│

├── utils/

│ └── preprocess.py

│

├── models/

│ ├── sales\_classifier.pkl

│ ├── feature\_order.pkl

│ ├── sales\_regressor.pkl

│ ├── reg\_feature\_order.pkl

│ ├── sales\_score\_p05.pkl

│ └── sales\_score\_p95.pkl

│

├── data/

│ └── sample\_product\_reviews.csv

│

└── .gitignore

