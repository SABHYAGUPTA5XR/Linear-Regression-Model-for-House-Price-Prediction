import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -- Page Config --
st.set_page_config(
    page_title="🏠 House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Theme Picker & CSS Injection --
theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Light"])
# Define colors for themes
if theme == "Light":
    bg_color = "#BADDD4"
    text_color = "#583000"
    px_template = "plotly_white"
else:
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    px_template = "plotly_dark"

# Inject CSS to override Streamlit default
st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .css-1d391kg {{ background-color: {bg_color}; }}  /* Sidebar container */
    .css-10trblm {{ color: {text_color}; }}         /* Primary header text */
    .css-1lcbmhc p {{ color: {text_color}; }}       /* Paragraph text */
</style>
""", unsafe_allow_html=True)

# -- Data Generation --
def generate_house_data(n_samples=200):
    np.random.seed(42)
    size = np.random.normal(1400, 50, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    age = np.random.uniform(0, 30, n_samples)
    # price formula + noise
    price = (
        size * 50 +
        bedrooms * 30000 -
        age * 1000 +
        np.random.normal(0, 5000, n_samples)
    )
    return pd.DataFrame({'size': size, 'bedrooms': bedrooms, 'age': age, 'price': price})

# -- Model Training / Loading --
MODEL_FILE = "house_model.pkl"

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    df = generate_house_data(n_samples=500)
    X = df[['size', 'bedrooms', 'age']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)

    # Compute metrics
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    return model, X_test, y_test, preds, mse, rmse, r2

# -- Main App --
model, X_test, y_test, preds, mse, rmse, r2 = load_or_train_model()

st.title("🏠 Linear Regression House Price Predictor")

tabs = st.tabs(["Predict", "Model Insights"])

# -- Tab 1: Prediction --
with tabs[0]:
    st.header("Predict the Price of Your House")
    col1, col2, col3 = st.columns(3)
    size = col1.slider('Size (sq ft)', 500, 3000, 1400)
    bedrooms = col2.slider('Bedrooms', 1, 6, 3)
    age = col3.slider('Age (years)', 0, 50, 10)

    if st.button('Predict Price'):
        input_df = pd.DataFrame({
            'size': [size],
            'bedrooms': [bedrooms],
            'age': [age]
        })
        price_pred = model.predict(input_df)[0]
        st.success(f'Predicted Price: ${price_pred:,.2f}')

        # Visualization with prediction
        df_vis = generate_house_data()
        fig = px.scatter(
            df_vis, x='size', y='price', color='age', size='bedrooms',
            title='House Price vs Size (colored by Age, sized by Bedrooms)',
            template=px_template
        )
        fig.add_scatter(
            x=[size], y=[price_pred], mode='markers', name='Your House',
            marker=dict(color='red', size=15, symbol='x')
        )
        st.plotly_chart(fig, use_container_width=True)

# -- Tab 2: Model Insights --
with tabs[1]:
    st.header("Model Performance Metrics")
    st.metric("MSE", f"{mse:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

    st.subheader("Residuals Distribution")
    resid_df = pd.DataFrame({'residual': y_test - preds})
    fig_resid = px.histogram(
        resid_df, x='residual', nbins=30,
        title='Histogram of Residuals',
        template=px_template
    )
    st.plotly_chart(fig_resid, use_container_width=True)

    st.subheader("Actual vs Predicted Prices")
    act_pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
    fig_line = px.scatter(
        act_pred_df, x='Actual', y='Predicted',
        title='Actual vs Predicted', template=px_template
    )
    fig_line.add_shape(
        type='line', x0=act_pred_df['Actual'].min(), y0=act_pred_df['Actual'].min(),
        x1=act_pred_df['Actual'].max(), y1=act_pred_df['Actual'].max(),
        line=dict(dash='dash')
    )
    st.plotly_chart(fig_line, use_container_width=True)

# -- Footer --
st.markdown("---")
st.write("Built with ❤️ using Streamlit, scikit-learn, and Plotly")
