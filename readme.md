# 🏠 House Price Predictor

A Streamlit web app that uses a Linear Regression model to predict house prices based on **size**, **bedrooms**, and **age**. The app features an interactive UI with light/dark themes, real-time predictions, and detailed model performance insights.

---

## Features

* **Interactive Prediction**: Adjust house parameters via sliders and get an instant price estimate.
* **Theme Switcher**: Toggle between Light and Dark modes to suit your preference.
* **Model Persistence**: Trains once and caches the model file (`house_model.pkl`) for fast subsequent loads.
* **Performance Metrics**: View MSE, RMSE, and R² scores on a held-out test set.
* **Data Visualizations**:

  * Scatter plot of all synthetic data points, colored by age and sized by bedrooms.
  * Highlight your specific input on the scatter.
  * Histogram of residuals.
  * Actual vs. Predicted scatter with identity line.

---

## 📦 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/house-price-predictor.git
   cd house-price-predictor
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add `.gitignore`** (if not already present)

   ```text
   __pycache__/
   *.py[cod]
   venv/
   .env/
   house_model.pkl
   ```

---

## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

* Use the **Predict** tab to input house features and get a price forecast.
* Switch to **Model Insights** to inspect metrics and visualizations of model performance.

---

## 📈 Data Generation

Uses a synthetic data generator (`generate_house_data`) with:

* `size ~ Normal(1400, 50)`
* `bedrooms ~ Uniform integers [1,5]`
* `age ~ Uniform(0,30)`
* Price formula: 

  ```text
  price = size*50 + bedrooms*30000 - age*1000 + noise(0,5000)
  ```

You can adjust `n_samples` or the distribution parameters in `app.py` to create more or varied data.

---

## 🛠️ Requirements

* Python 3.8+
* streamlit
* numpy
* pandas
* scikit-learn
* plotly
* joblib

All versions are pinned in `requirements.txt`.

---

## 📦 Deployment

1. Push your code to GitHub.
2. Connect the repo on Streamlit Cloud ([https://share.streamlit.io](https://share.streamlit.io)).
3. Select `main` branch and `app.py` file.
4. Deploy and share your dashboard!

---

## ❤️ Contributing

Feel free to open issues or PRs for:

* Adding real-world datasets
* More sophisticated models (e.g., Random Forest, XGBoost)
* Enhanced UI/UX (charts, filters, user authentication)

---

*Built with ❤️ by Sabhya Gupta*
