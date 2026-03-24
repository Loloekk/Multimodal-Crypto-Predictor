import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Softmax
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers.models.auto.modeling_tf_auto import TFAutoModelForSequenceClassification
from transformers.utils import logging
logging.set_verbosity_error()
import kagglehub


SYMBOL = "BTCUSDT"
INTERVAL = "1h"
START_DATE = "12 Oct 2021"
END_DATE = "12 Sep 2024"

SEQ_LEN = 72
SENT_LEN = 12
HORIZON = 6
THRESHOLD = 0.0001
SEED = 42
MAX_TEXT_LEN = 512 

np.random.seed(SEED)
tf.random.set_seed(SEED)



kaggle_path = kagglehub.dataset_download("imadallal/sentiment-analysis-of-bitcoin-news-2021-2024")
csv_path = "/home/z1203964/.cache/kagglehub/datasets/imadallal/sentiment-analysis-of-bitcoin-news-2021-2024/versions/2/bitcoin_sentiments_21_24.csv"

def date_to_milliseconds(date_str):
    dt = datetime.strptime(date_str, "%d %b %Y")
    return int(dt.timestamp() * 1000)

def fetch_binance_klines(symbol, interval, start_str, end_str):
    url = "https://api.binance.com/api/v3/klines"
    start_ts = date_to_milliseconds(start_str)
    end_ts = date_to_milliseconds(end_str)

    all_data = []
    limit = 1000

    while start_ts < end_ts:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ts, "endTime": end_ts, "limit": limit}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        all_data.extend(data)
        start_ts = int(data[-1][0]) + 1

    btc_df = pd.DataFrame(all_data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    btc_df["open_time"] = pd.to_datetime(btc_df["open_time"], unit="ms")
    btc_df.set_index("open_time", inplace=True)
    
    for col in ["open","high","low","close","volume"]:
        btc_df[col] = pd.to_numeric(btc_df[col], errors="coerce")
    return btc_df[["open","high","low","close","volume"]]

btc_df = fetch_binance_klines(SYMBOL, INTERVAL, START_DATE, END_DATE)

btc_df["return"] = btc_df["close"].pct_change()
btc_df["rsi"] = ta.momentum.RSIIndicator(btc_df["close"], window=14).rsi()
btc_df["ema_fast"] = ta.trend.EMAIndicator(btc_df["close"], window=12).ema_indicator()
btc_df["ema_slow"] = ta.trend.EMAIndicator(btc_df["close"], window=26).ema_indicator()
btc_df["macd"] = ta.trend.MACD(btc_df["close"]).macd()
btc_df["atr"] = ta.volatility.AverageTrueRange(btc_df["high"], btc_df["low"], btc_df["close"]).average_true_range()
btc_df.dropna(inplace=True)



sent_df = pd.read_csv(csv_path)
date_col = "Date"
text_col = "Short Description"

sent_df[date_col] = pd.to_datetime(sent_df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
sent_df.dropna(subset=[date_col], inplace=True)
sent_df.set_index(date_col, inplace=True)


sent_hourly_text = sent_df.groupby(pd.Grouper(freq='1h'))[text_col].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
sent_hourly_text.set_index(date_col, inplace=True)

btc_df = btc_df.merge(sent_hourly_text, left_index=True, right_index=True, how="left")
btc_df[text_col] = btc_df[text_col].fillna("No news")


text_series = btc_df[text_col].values
rolling_texts = []
for i in range(len(text_series)):
    start_idx = max(0, i - SENT_LEN + 1)
    combined_text = " ".join(text_series[start_idx : i+1])
    rolling_texts.append(combined_text)

btc_df['rolling_text'] = rolling_texts

btc_df["future_close"] = btc_df["close"].shift(-HORIZON)
btc_df["future_return"] = (btc_df["future_close"] - btc_df["open"]) / btc_df["open"]
btc_df["target"] = np.where(btc_df["future_return"] > THRESHOLD, 1, np.where(btc_df["future_return"] < -THRESHOLD, -1, 0))
btc_df.dropna(inplace=True)


FEATURES_SEQ = ["return", "rsi", "ema_fast", "ema_slow", "macd", "atr", "volume"]
split_idx = int(0.8 * len(btc_df))

train_df = btc_df.iloc[:split_idx].copy()
test_df = btc_df.iloc[split_idx:].copy()

scaler_seq = StandardScaler()
X_train_seq_scaled = scaler_seq.fit_transform(train_df[FEATURES_SEQ])
X_test_seq_scaled = scaler_seq.transform(test_df[FEATURES_SEQ])


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors="np"
    )

train_encodings = tokenize_texts(train_df['rolling_text'])
test_encodings = tokenize_texts(test_df['rolling_text'])

y_train_raw = train_df["target"].values
y_test_raw = test_df["target"].values

def make_multi_inputs(X_seq, input_ids_all, attention_mask_all, y, seq_len):
    Xs_seq, Xs_ids, Xs_masks, ys = [], [], [], []
    for i in range(seq_len, len(X_seq)):
        Xs_seq.append(X_seq[i-seq_len:i])
        Xs_ids.append(input_ids_all[i])
        Xs_masks.append(attention_mask_all[i])
        ys.append(y[i])
    return np.array(Xs_seq), np.array(Xs_ids), np.array(Xs_masks), np.array(ys)

X_train_seq, X_train_ids, X_train_masks, y_train = make_multi_inputs(
    X_train_seq_scaled, train_encodings['input_ids'], train_encodings['attention_mask'], y_train_raw, SEQ_LEN)

X_test_seq, X_test_ids, X_test_masks, y_test = make_multi_inputs(
    X_test_seq_scaled, test_encodings['input_ids'], test_encodings['attention_mask'], y_test_raw, SEQ_LEN)

y_train = np.where(y_train == -1, 0, np.where(y_train == 0, 1, 2))
y_test = np.where(y_test == -1, 0, np.where(y_test == 0, 1, 2))


input_seq = Input(shape=(SEQ_LEN, len(FEATURES_SEQ)), name="ts_input")
x = LSTM(64, return_sequences=True)(input_seq)
x = Dropout(0.3)(x)
x = LSTM(32)(x)
x = Dropout(0.3)(x)
lstm_out = Dense(16, activation="relu")(x)

input_ids = Input(shape=(MAX_TEXT_LEN,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(MAX_TEXT_LEN,), dtype=tf.int32, name="attention_mask")

finbert_layer = TFAutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

finbert_layer.trainable = True
for layer in finbert_layer.layers:
    layer.trainable = False

finbert_layer.get_layer("classifier").trainable = True

bert_outputs = finbert_layer(input_ids, attention_mask=attention_mask)[0]
finbert_probs = Softmax(name="sentiment_probs")(bert_outputs)

merged = Concatenate()([lstm_out, finbert_probs])

d1 = Dense(32, activation="relu")(merged)
d2 = Dropout(0.3)(d1)
d3 = Dense(16, activation="relu")(d2)
output = Dense(3, activation="softmax")(d3)

model = Model(inputs=[input_seq, input_ids, attention_mask], outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

model.compile(
    optimizer=optimizer, 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

unique_classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(unique_classes, cw)}

history = model.fit(
    [X_train_seq, X_train_ids, X_train_masks], y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    verbose=1
)


input_seq_only = Input(shape=(SEQ_LEN, len(FEATURES_SEQ)), name="ts_input_only")

x2 = LSTM(64, return_sequences=True)(input_seq_only)
x2 = Dropout(0.3)(x2)
x2 = LSTM(32)(x2)
x2 = Dropout(0.3)(x2)
x2 = Dense(16, activation="relu")(x2)

output_only = Dense(3, activation="softmax")(x2)

model_lstm_only = Model(inputs=input_seq_only, outputs=output_only)

model_lstm_only.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_lstm = model_lstm_only.fit(
    X_train_seq,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    verbose=1
)


print("\n--- WYNIKI EWALUACJI ---")
print("LSTM + FinBERT (Multimodalny):")
loss, accuracy = model.evaluate([X_test_seq, X_test_ids, X_test_masks], y_test, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

print("\nTylko LSTM:")
loss_lstm, acc_lstm = model_lstm_only.evaluate(X_test_seq, y_test, verbose=0)
print(f"Loss: {loss_lstm:.4f}")
print(f"Accuracy: {acc_lstm:.4f}")


preds_multi = np.argmax(model.predict([X_test_seq, X_test_ids, X_test_masks], verbose=0), axis=1)
preds_lstm = np.argmax(model_lstm_only.predict(X_test_seq, verbose=0), axis=1)

bt_df = btc_df.iloc[split_idx:split_idx + len(y_test)].copy()
bt_closes = bt_df['close'].values

def run_backtest(predictions, closes, horizon):
    capital = 1.0
    equity_curve = np.ones(len(closes))
    in_trade_until = -1

    for i in range(len(predictions)):
        equity_curve[i] = capital

        if i < in_trade_until:
            continue

        signal = predictions[i]

        if signal != 1 and (i + horizon) < len(closes):
            if signal == 2: 
                trade_return = (closes[i + horizon] - closes[i]) / closes[i]
            elif signal == 0: 
                trade_return = (closes[i] - closes[i + horizon]) / closes[i]

            in_trade_until = i + horizon
            capital *= (1 + trade_return)

    for j in range(max(0, in_trade_until), len(closes)):
        if j < len(closes):
            equity_curve[j] = capital

    return equity_curve

equity_multi = run_backtest(preds_multi, bt_closes, HORIZON)
equity_lstm = run_backtest(preds_lstm, bt_closes, HORIZON)

bt_df['Multi'] = equity_multi
bt_df['LSTM'] = equity_lstm


plt.figure(figsize=(14, 8))

plt.plot(bt_df.index, bt_df['LSTM'],
         label='Tylko LSTM',
         linewidth=1.5, color='blue')

plt.plot(bt_df.index, bt_df['Multi'],
         label='LSTM + FinBERT (End-to-End)',
         linewidth=1.5, color='orange')

plt.title('Porównanie krzywych kapitału (Equity Curve)', fontsize=14)
plt.xlabel('Czas', fontsize=12)
plt.ylabel('Mnożnik kapitału początkowego', fontsize=12)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('result.png', dpi=300)
print("\nWykres został zapisany jako 'result.png'.")
plt.show()