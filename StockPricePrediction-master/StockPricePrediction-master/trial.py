import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from langdetect import detect
import os
import base64

# Cache the sentiment models and pipelines
@st.cache_resource
def load_sentiment_models():
    french_tokenizer = AutoTokenizer.from_pretrained('ac0hik/Sentiment_Analysis_French')
    french_model = AutoModelForSequenceClassification.from_pretrained('ac0hik/Sentiment_Analysis_French')
    
    arabic_tokenizer = AutoTokenizer.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment')
    arabic_model = AutoModelForSequenceClassification.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment')
    
    return french_tokenizer, french_model, arabic_tokenizer, arabic_model

french_tokenizer, french_model, arabic_tokenizer, arabic_model = load_sentiment_models()

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file, parse_dates=['Date'])

def handle_missing_values(series, n_days):
    last_value = 1
    count = 0
    result = []
    for value in series:
        if not np.isnan(value) and value != 0:
            last_value = value
            count = 0
        result.append(last_value if (value == 0 or np.isnan(value)) and count < n_days else value)
        count = count + 1 if (value == 0 or np.isnan(value)) else 0
    return result

def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size, 0])
    return np.array(X), np.array(y)

def analyze_sentiment_adaptive(text, language):
    if language == 'fr':
        tokenizer = french_tokenizer
        model = french_model
    elif language == 'ar':
        tokenizer = arabic_tokenizer
        model = arabic_model
    else:
        return [{'label': 'neutral', 'score': 0}]
    
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=510)

    if len(tokens['input_ids'][0]) <= 512:
        outputs = model(**tokens)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = scores.detach().numpy()[0]
        label = 'positive' if scores[1] > scores[0] else 'negative'
        return [{'label': label, 'score': max(scores)}]
    else:
        max_length = 510
        stride = 200
        all_scores = []

        for i in range(0, len(tokens['input_ids'][0]), stride):
            batch = tokens['input_ids'][0][i:i+max_length]
            batch = torch.unsqueeze(batch, 0)
            outputs = model(batch)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_scores.append(scores.detach().numpy()[0])

        avg_scores = np.mean(all_scores, axis=0)
        label = 'positive' if avg_scores[1] > avg_scores[0] else 'negative'
        return [{'label': label, 'score': max(avg_scores)}]

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'  # Fallback to 'unknown' if detection fails

def absolute_sentiment(s):
    label = s[0]['label']
    if label == 'positive':
        return 'positive', 1
    elif label == 'negative':
        return 'negative', -1
    else:
        return 'neutral', 0

def train_model(data, company, include_sentiment, n_days, window_size):
    K.clear_session()
    tf.keras.utils.set_random_seed(0)
    
    company_data = data[data['Company'] == company].copy()
    company_data.set_index('Date', inplace=True)
    company_data.sort_index(inplace=True)
    
    if include_sentiment:
        company_data['TitleLanguage'] = company_data['Title'].apply(detect_language)
        company_data['ContentLanguage'] = company_data['Content'].apply(detect_language)
        
        company_data['TitleSentiment'] = company_data.apply(lambda row: analyze_sentiment_adaptive(row['Title'], row['TitleLanguage']), axis=1)
        company_data['ContentSentiment'] = company_data.apply(lambda row: analyze_sentiment_adaptive(row['Content'], row['ContentLanguage']), axis=1)
        
        company_data['TitleSentimentLabel'], company_data['TitleSentiment'] = zip(*company_data['TitleSentiment'].apply(absolute_sentiment))
        company_data['ContentSentimentLabel'], company_data['ContentSentiment'] = zip(*company_data['ContentSentiment'].apply(absolute_sentiment))
        
        company_data['TitleSentiment'] = handle_missing_values(company_data['TitleSentiment'].values, n_days)
        company_data['ContentSentiment'] = handle_missing_values(company_data['ContentSentiment'].values, n_days)
        
        features = company_data[['Cours ajusté', 'ContentSentiment', 'TitleSentiment']].values
    else:
        features = company_data[['Cours ajusté']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    X, y = create_dataset(features_scaled, window_size)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.1)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    model = Sequential([
        GRU(units=64, return_sequences=True, input_shape=(window_size, X.shape[2]), recurrent_initializer='glorot_normal'),
        GRU(units=64, return_sequences=True, recurrent_initializer='glorot_normal'),
        GRU(units=32, recurrent_initializer='glorot_normal'),
        Dense(16, kernel_initializer='glorot_normal'),
        Dense(1, kernel_initializer='glorot_normal')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=8,
                        validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler.inverse_transform(np.concatenate([y_pred, np.zeros((y_pred.shape[0], features.shape[1] - 1))], axis=1))[:, 0]
    y_test_inverse = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], features.shape[1] - 1))], axis=1))[:, 0]
    
    company_data['Cours ajusté'] = scaler.inverse_transform(np.concatenate([features_scaled[:, [0]], np.zeros((features_scaled.shape[0], features.shape[1] - 1))], axis=1))[:, 0]
    
    # Save the model and scaler for future use
    model_filename = f'{company}_model{"_with_sentiment" if include_sentiment else ""}.h5'
    scaler_filename = f'{company}_scaler{"_with_sentiment" if include_sentiment else ""}.gz'
    save_model(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    # Save to session state
    st.session_state[f'{company}_model{"_with_sentiment" if include_sentiment else ""}'] = model
    st.session_state[f'{company}_scaler{"_with_sentiment" if include_sentiment else ""}'] = scaler
    
    return company_data, y_test_inverse, y_pred_inverse, history

def plot_results(company_data, y_test_inverse, y_pred_inverse, history):
    st.subheader('Training and Validation Loss')
    fig = px.line(x=range(len(history.history['loss'])), y=[history.history['loss'], history.history['val_loss']],
                  labels={'x': 'Epoch', 'value': 'Loss'}, title='Training and Validation Loss')
    fig.data[0].name = 'Training Loss'
    fig.data[1].name = 'Validation Loss'
    fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss', colorway=["#636EFA", "#EF553B"])
    st.plotly_chart(fig)
    
    st.subheader('Stock Price Prediction')
    fig = px.line(company_data, x=company_data.index, y='Cours ajusté', title='Stock Price Prediction with GRU and Sentiment Analysis')
    fig.add_scatter(x=company_data.index[-len(y_test_inverse):], y=y_pred_inverse, mode='lines', name='GRU Predictions')
    fig.update_layout(xaxis_title='Date', yaxis_title='Adjusted Closing Price', colorway=["#00CC96", "#AB63FA"])
    st.plotly_chart(fig)

def predict_next_day(company, latest_data, window_size, include_sentiment):
    model_key = f'{company}_model{"_with_sentiment" if include_sentiment else ""}'
    scaler_key = f'{company}_scaler{"_with_sentiment" if include_sentiment else ""}'
    
    if model_key not in st.session_state or scaler_key not in st.session_state:
        st.error("Model not found. Please train the model first.")
        return None
    
    model = st.session_state[model_key]
    scaler = st.session_state[scaler_key]
    
    if include_sentiment:
        latest_data = latest_data[['Cours ajusté', 'ContentSentiment', 'TitleSentiment']].values
    else:
        latest_data = latest_data[['Cours ajusté']].values
    
    latest_data_scaled = scaler.transform(latest_data)
    X_latest = np.reshape(latest_data_scaled, (1, window_size, latest_data_scaled.shape[1]))
    
    next_day_pred_scaled = model.predict(X_latest)
    next_day_pred = scaler.inverse_transform(np.concatenate([next_day_pred_scaled, np.zeros((1, latest_data.shape[1] - 1))], axis=1))[:, 0]
    
    return next_day_pred[0]

def training_page():
    st.title("Stock Price Prediction - Model Training")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="training_file_uploader")
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.session_state['data'] = data
    
    if 'data' in st.session_state:
        data = st.session_state['data']
        companies = data['Company'].unique()
        
        selected_company = st.selectbox("Select Company", companies, key="training_company_selectbox")
        include_sentiment = st.checkbox("Include Sentiment in Training Features", value=True, key="training_sentiment_checkbox")
        n_days = 7
        if include_sentiment:
            n_days = st.slider("Number of Days to Handle Missing Values", 1, 30, 7, key="training_n_days_slider")
        
        window_size = st.slider("Window Size", 1, 50, 5, key="training_window_size_slider")
        
        if st.button("Train Model", key="train_model_button"):
            st.write("Training the model...")
            progress_bar = st.progress(0)
            
            company_data, y_test_inverse, y_pred_inverse, history = train_model(data, selected_company, include_sentiment, n_days, window_size)
            
            progress_bar.progress(100)
            st.write("Model training completed.")
            
            plot_results(company_data, y_test_inverse, y_pred_inverse, history)
            
            mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
            mse = mean_squared_error(y_test_inverse, y_pred_inverse)
            rmse = mse ** 0.5
            
            st.write(f'Mean Absolute Error (MAE): {mae}')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')

def prediction_page():
    st.title("Stock Price Prediction - Next Day Prediction")
    selected_company = st.selectbox("Select Company", ['AFMA', 'AGMA', 'ATLANTASANAD', 'SANLAM MAROC', 'WAFA ASSURANCE'], key="prediction_company_selectbox")
    include_sentiment = st.checkbox("Include Sentiment in Prediction", value=True, key="prediction_sentiment_checkbox")
    window_size = st.slider("Window Size", 1, 50, 5, key="prediction_window_size_slider")
    
    adjusted_closing_price = st.number_input("Enter Today's Adjusted Closing Price", value=0.0, key="prediction_adjusted_closing_price")
    
    if include_sentiment:
        news_title = st.text_input("Enter Today's News Title", key="prediction_news_title")
        news_content = st.text_area("Enter Today's News Content", key="prediction_news_content")

        title_language = detect_language(news_title)
        content_language = detect_language(news_content)
        
        title_sentiment = analyze_sentiment_adaptive(news_title, title_language)
        content_sentiment = analyze_sentiment_adaptive(news_content, content_language)
        
        title_sentiment_label, title_sentiment_score = absolute_sentiment(title_sentiment)
        content_sentiment_label, content_sentiment_score = absolute_sentiment(content_sentiment)
        
        st.write(f"Title Sentiment: {title_sentiment_label}")
        st.write(f"Content Sentiment: {content_sentiment_label}")
    else:
        title_sentiment_score = 0
        content_sentiment_score = 0
    
    if st.button("Predict Next Day", key="predict_next_day_button"):
        latest_data = pd.DataFrame({
            'Cours ajusté': [adjusted_closing_price] * window_size
        })
        if include_sentiment:
            latest_data['ContentSentiment'] = [content_sentiment_score] * window_size
            latest_data['TitleSentiment'] = [title_sentiment_score] * window_size
        
        next_day_prediction = predict_next_day(selected_company, latest_data, window_size, include_sentiment)
        if next_day_prediction is not None:
            st.write(f"Predicted Stock Price for Tomorrow: {next_day_prediction}")

def explore_data_page():
    st.title("Explore Adjusted Closing Stock Price")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="explore_file_uploader")
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.session_state['explore_data'] = data  # Store the data in session state

    if 'explore_data' in st.session_state:
        data = st.session_state['explore_data']
        companies = data['Company'].unique()
        
        selected_company = st.selectbox("Select Company", companies, key="explore_company_selectbox")
        
        if st.button("Display Results", key="display_results_button"):
            explore_data(data, selected_company)

def explore_data(data, selected_company):
    company_data = data[data['Company'] == selected_company]
    
    st.subheader(f"Descriptive Statistics for {selected_company}")
    company_stats = company_data['Cours ajusté'].describe()
    st.write(company_stats)
    
    st.subheader(f"Adjusted Closing Stock Price of {selected_company} Over Time")
    fig = px.line(company_data, x='Date', y='Cours ajusté', title=f'Adjusted Closing Stock Price of {selected_company} Over Time')
    fig.update_layout(xaxis_title='Date', yaxis_title='Adjusted Closing Price', colorway=["#636EFA"])
    st.plotly_chart(fig)
    
    st.subheader(f"Distribution of Adjusted Closing Price for {selected_company}")
    hist_data = [company_data['Cours ajusté']]
    group_labels = ['Adjusted Closing Price']  # name of the dataset
    fig = ff.create_distplot(hist_data, group_labels, bin_size=1.0, show_rug=False)
    fig.update_layout(title=f'Distribution of Adjusted Closing Price for {selected_company}', xaxis_title='Adjusted Closing Price', yaxis_title='Density', colorway=["#EF553B"])
    st.plotly_chart(fig)
    
    st.subheader(f"Adjusted Closing Price of {selected_company} by Year")
    company_data['Year'] = company_data['Date'].dt.year
    fig = px.box(company_data, x='Year', y='Cours ajusté', title=f'Adjusted Closing Price of {selected_company} by Year')
    fig.update_layout(xaxis_title='Year', yaxis_title='Adjusted Closing Price', colorway=["#00CC96"])
    st.plotly_chart(fig)
    
    st.set_page_config(layout="wide")
    
def home_page():
    html_url = "http://localhost:8000/index.html"
    # Inject CSS to modify width properties
    css_style = """
  <style>
        /* Full width adjustments */
        .stApp, .block-container, .element-container {
            width: 100% !important;
            max-width: 100% !important;
            padding: 0 !important;
        }
        /* Ensure iframe takes full width and reasonable height */
       
        /* Additional tweaks for layout improvements */
        .css-1adrfps {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
    </style>
    """
    st.markdown(
        f'<iframe src="{html_url}" style="width:100%;height:800px;border:none;" seamless></iframe>',
        unsafe_allow_html=True
    )
    st.markdown(css_style, unsafe_allow_html=True)


def main():
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Go to", ["Home", "Explore Stock Price Data", "Train Prediction Model", "Predict Next Day"], key="page_selectbox")

    if page == "Home":
        home_page()
    else:
        with st.container():
            st.markdown('<div class="styled-page">', unsafe_allow_html=True)
            if page == "Explore Stock Price Data":
                explore_data_page()
            elif page == "Train Prediction Model":
                training_page()
            elif page == "Predict Next Day":
                prediction_page()
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
