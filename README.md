# Multimodal Crypto Predictor: LSTM vs LSTM+FinBERT

This project predicts short-term Bitcoin (BTCUSDT) price movements (Up, Down, Neutral) over a 6-hour horizon.

I build and compare two models:
1. **LSTM-only:** Baseline using only technical indicators (RSI, MACD, EMA, ATR).
2. **Multimodal Model (LSTM + FinBERT):** Combines numerical market data with news sentiment.

### Data Pipeline (Downloading)
* **Market data:** Downloaded dynamically on the fly via the Binance API . No need to manage manual CSV files.
* **News data:** Fetched automatically using the `kagglehub` library directly from the Kaggle dataset (`imadallal/sentiment-analysis-of-bitcoin-news-2021-2024`).

### Transfer Learning & Fine-Tuning
The project leverages **Transfer Learning** to process text. Instead of training a language model from scratch, we import `ProsusAI/finbert` from HuggingFace, which is already pre-trained on financial texts. 
For **Fine-Tuning**, we freeze the heavy BERT base layers (to retain its financial knowledge and save compute power) and only unfreeze the final classification head. This head is trained end-to-end alongside our LSTM to adapt specifically to our BTC prediction task.

### Technologies
* **Python**
* **TensorFlow / Keras** (Deep Learning)
* **Transformers / HuggingFace** (FinBERT)
* **Pandas & NumPy** (Data manipulation)
* **Scikit-learn** (Data scaling and class weights)
* **ta** (Technical Analysis library)

---
**Author:** Karol Bielaszka  
Project created for the Artificial Intelligence course in the Theoretical Computer Science program at Jagiellonian University (UJ).
