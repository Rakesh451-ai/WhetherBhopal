import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Weather Forecaster",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CACHING DATA & MODEL (For Speed)
# ==========================================
@st.cache_data
def load_background_data():
    """Generates the 10-year actual dataset (2020-2030) for evaluation."""
    np.random.seed(101)
    days = 365 * 11 
    time = np.arange(days)
    base_temp = 25.5 
    
    yearly_season = 8 * np.sin(2 * np.pi * (time - 100) / 365) 
    monsoon_drop = -4 * np.exp(-(((time % 365) - 200) / 40)**2) 
    winter_trough = -5 * np.cos(2 * np.pi * time / 365)
    
    ar_noise = np.zeros(days)
    ar_noise[0] = np.random.normal(0, 1)
    for t in range(1, days):
        ar_noise[t] = 0.6 * ar_noise[t-1] + np.random.normal(0, 1.5)

    temperature = base_temp + yearly_season + monsoon_drop + winter_trough + ar_noise
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    df_daily = pd.DataFrame({'Date': dates, 'Actual_Temp': temperature})
    df_daily.set_index('Date', inplace=True)
    
    df_monthly = df_daily.resample('ME').mean()
    return df_daily, df_monthly

@st.cache_resource
def load_model():
    """Loads the trained SARIMA model."""
    file_name = 'bhopal_sarima_verified.pkl'
    try:
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

df_daily, df_monthly_actuals = load_background_data()
model = load_model()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🌤️ Forecaster Menu")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", 
    ["Live Prediction Engine", "Interactive Dashboard", "Model Report Card", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.info("Powered by ARIMA Machine Learning\n\nO")

# ==========================================
# 4. PAGE 1: LIVE PREDICTION ENGINE
# ==========================================
if page == "Live Prediction Engine":
    st.title("Live Prediction Engine")
    st.markdown("Enter a month and year to generate a temperature forecast.")
    
    if model is None:
        st.error("🚨 Critical Error: `bhopal_sarima_verified.pkl` not found in the directory. Please upload your model file.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Target Date")
            selected_date = st.date_input("Date (Day will default to month-end)", min_value=datetime(2022, 1, 1))
            predict_btn = st.button("Generate Forecast", type="primary")
            
        if predict_btn:
            with st.spinner('Calculating atmospheric vectors...'):
                target_date = pd.to_datetime(selected_date)
                target_end_of_month = target_date + pd.offsets.MonthEnd(0)
                current_date = pd.to_datetime('today')
                
                # Fetch Prediction
                forecast = model.predict(start=target_end_of_month, end=target_end_of_month)
                predicted_temp = forecast.iloc[0]
                
                with col2:
                    st.subheader(f"Results for {target_date.strftime('%B %Y')}")
                    
                    # Logic Router for Past vs Future
                    if target_end_of_month <= current_date and target_end_of_month in df_monthly_actuals.index:
                        actual_temp = df_monthly_actuals.loc[target_end_of_month, 'Actual_Temp']
                        error_margin = abs(actual_temp - predicted_temp)
                        mape = (error_margin / actual_temp) * 100
                        
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Predicted Temp", f"{predicted_temp:.2f} °C")
                        m_col2.metric("Actual Temp", f"{actual_temp:.2f} °C", delta=f"{predicted_temp - actual_temp:.2f} °C", delta_color="inverse")
                        m_col3.metric("Error Margin", f"{mape:.2f} %")
                        
                        if mape < 5:
                            st.success("🟢 Outstanding Accuracy: Model performed exceptionally well.")
                        elif mape < 15:
                            st.warning("🟡 Acceptable Variance: Model is within normal bounds.")
                        else:
                            st.error("🔴 High Error: Weather anomalies detected.")
                            
                    else:
                        st.metric("Future Predicted Temp", f"{predicted_temp:.2f} °C")
                        st.info("🔮 Future Forecast Mode: This date has not occurred yet, so error metrics cannot be calculated.")

# ==========================================
# 5. PAGE 2: INTERACTIVE DASHBOARD
# ==========================================
elif page == "Interactive Dashboard":
    st.title("Interactive Weather Analytics")
    st.markdown("Analyze a 30-day simulated weather window using Plotly.")
    
    # Grab the last 30 days from our generated daily dataset
    df_30 = df_daily.tail(30).copy()
    df_30['Week'] = df_30.index.isocalendar().week
    
    def categorize_temp(t):
        if t < 25: return 'Cool (<25°C)'
        elif 25 <= t <= 32: return 'Warm (25-32°C)'
        else: return 'Hot (>32°C)'
    df_30['Category'] = df_30['Actual_Temp'].apply(categorize_temp)
    
    # Plotly Dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("30-Day Temperature Trend", "Temperature Distribution",
                        "Average Temperature by Week", "Weather Categories"),
        specs=[[{"type": "xy"}, {"type": "histogram"}],
               [{"type": "xy"}, {"type": "domain"}]]
    )

    fig.add_trace(go.Scatter(x=df_30.index, y=df_30['Actual_Temp'], mode='lines+markers', name='Daily Temp', line=dict(color='#ff5722')), row=1, col=1)
    fig.add_trace(go.Histogram(x=df_30['Actual_Temp'], nbinsx=10, name='Days', marker_color='#03a9f4'), row=1, col=2)
    
    weekly_avg = df_30.groupby('Week')['Actual_Temp'].mean().reset_index()
    fig.add_trace(go.Bar(x=weekly_avg['Week'], y=weekly_avg['Actual_Temp'], name='Weekly Avg', marker_color='#4caf50'), row=2, col=1)
    
    cat_counts = df_30['Category'].value_counts()
    fig.add_trace(go.Pie(labels=cat_counts.index, values=cat_counts.values, marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99']), hole=0.3), row=2, col=2)

    fig.update_layout(height=700, showlegend=False, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. PAGE 3: MODEL REPORT CARD
# ==========================================
elif page == "Model Report Card":
    st.title("Model Performance Report")
    st.markdown("Evaluating the ARIMA mathematical parameters on holdout data.")
    
    col1, col2, col3 = st.columns(3)
    # Using the metrics we calculated earlier in the project
    col1.metric("RMSE", "1.42 °C", help="Root Mean Squared Error")
    col2.metric("MAE", "1.15 °C", help="Mean Absolute Error")
    col3.metric("MAPE", "4.85 %", help="Mean Absolute Percentage Error")
    
    st.markdown("### Technical Architecture")
    st.code("""
    Model Algorithm: Seasonal AutoRegressive Integrated Moving Average (SARIMA)
    Base Configuration (p,d,q): (1, 1, 1)
    Seasonal Configuration (P,D,Q,s): (1, 1, 1, 12)
    Optimization: Monthly Resampling for compute efficiency
    """, language="text")

# ==========================================
# 7. PAGE 4: ABOUT
# ==========================================
elif page == "About Project":
    st.title("About This Project")
    st.markdown("""
    ### Weather Forecasting System
    This is Weather Forecaster predictior model . 
    
    **Objective:** To demonstrate an end-to-end machine learning data pipeline, from synthetic climate modeling and data engineering (resampling) to mathematical evaluation and front-end UI deployment.
    
    **Tools Used:**
    * **Backend:** Python, Pandas, NumPy
    * **Machine Learning:** Statsmodels (SARIMA), Scikit-Learn
    * **Frontend:** Streamlit, Plotly
    """)