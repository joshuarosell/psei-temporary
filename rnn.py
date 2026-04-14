import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────
LOOKBACK    = 10          # past trading days used as input sequence
BATCH_SIZE  = 32
MAX_EPOCHS  = 100
RANDOM_SEED = 42

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "positive_sent", "negative_sent", "neutral_sent", "sent_score",
    "has_disclosure",
    "sma_20", "ema_12", "ema_26",
    "macd", "signal_line", "macd_histogram",
    "rsi",
]
TARGET_COL   = "target_encoded"
CLASS_NAMES  = ["DOWN", "FLAT", "UP"]   # 0 → DOWN, 1 → FLAT, 2 → UP

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────
# 2. Load data
# ──────────────────────────────────────────────
train_df = pd.read_csv("train_data.csv")
val_df   = pd.read_csv("val_data.csv")
test_df  = pd.read_csv("test_data.csv")

# ──────────────────────────────────────────────
# 3. Scale features — fit ONLY on training set
# ──────────────────────────────────────────────
scaler = StandardScaler()
train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
val_df[FEATURE_COLS]   = scaler.transform(val_df[FEATURE_COLS])
test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

# ──────────────────────────────────────────────
# 4. Build sliding-window sequences
# ──────────────────────────────────────────────
def make_sequences(df, lookback):
    """
    Returns:
        X: (n_samples, lookback, n_features)
        y: (n_samples,)  integer class labels
    """
    X_vals = df[FEATURE_COLS].values
    y_vals = df[TARGET_COL].values.astype(int)

    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(X_vals[i - lookback : i])
        y.append(y_vals[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


X_train, y_train = make_sequences(train_df, LOOKBACK)
X_val,   y_val   = make_sequences(val_df,   LOOKBACK)
X_test,  y_test  = make_sequences(test_df,  LOOKBACK)

print(f"Train sequences : {X_train.shape}  labels: {y_train.shape}")
print(f"Val   sequences : {X_val.shape}  labels: {y_val.shape}")
print(f"Test  sequences : {X_test.shape}  labels: {y_test.shape}")

# ──────────────────────────────────────────────
# 5. Model builder for Keras Tuner
# ──────────────────────────────────────────────
n_features = len(FEATURE_COLS)
n_classes  = len(CLASS_NAMES)

def build_model(hp):
    units_1      = hp.Choice("lstm_units_1",  [32, 64, 128])
    units_2      = hp.Choice("lstm_units_2",  [16, 32, 64])
    dropout_rate = hp.Float("dropout_rate",   min_value=0.1, max_value=0.5, step=0.1)
    lr           = hp.Choice("learning_rate", [1e-4, 5e-4, 1e-3])

    m = Sequential([
        LSTM(units_1, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(dropout_rate),
        LSTM(units_2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(n_classes, activation="softmax"),
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m

# ──────────────────────────────────────────────
# 6. Class weights (address class imbalance)
# ──────────────────────────────────────────────
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(n_classes),
    y=y_train,
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
print(f"Class weights: {class_weight_dict}")

# ──────────────────────────────────────────────
# 7. Keras Tuner — hyperparameter search
# ──────────────────────────────────────────────
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_loss",
    max_trials=20,
    executions_per_trial=1,
    directory="kt_search",
    project_name="lstm_price_movement",
    overwrite=True,
    seed=RANDOM_SEED,
)

tuner.search_space_summary()

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    class_weight=class_weight_dict,
    verbose=1,
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n── Best Hyperparameters ──")
print(f"  lstm_units_1  : {best_hps.get('lstm_units_1')}")
print(f"  lstm_units_2  : {best_hps.get('lstm_units_2')}")
print(f"  dropout_rate  : {best_hps.get('dropout_rate')}")
print(f"  learning_rate : {best_hps.get('learning_rate')}")

# ──────────────────────────────────────────────
# 8. Retrain best model to convergence
# ──────────────────────────────────────────────
model = build_model(best_hps)
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    "model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weight_dict,
    verbose=1,
)

# ──────────────────────────────────────────────
# 9. Evaluate on test set
# ──────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

print("\n── Classification Report (Overall) ──")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ──────────────────────────────────────────────
# 10. Per-stock performance analysis
# ──────────────────────────────────────────────
# Align predictions with the test DataFrame rows used to build sequences.
# make_sequences drops the first `LOOKBACK` rows, so slice accordingly.
test_stocks = test_df["stock"].iloc[LOOKBACK:].reset_index(drop=True).values

stock_tickers = sorted(set(test_stocks))
per_stock_rows = []

print("\n── Per-Stock Performance ──")
for ticker in stock_tickers:
    mask        = test_stocks == ticker
    y_true_s    = y_test[mask]
    y_pred_s    = y_pred[mask]
    n_samples   = mask.sum()

    acc         = (y_true_s == y_pred_s).mean()
    per_stock_rows.append({"stock": ticker, "n_samples": n_samples, "accuracy": acc})

    print(f"\nStock: {ticker}  (n={n_samples})")
    print(classification_report(
        y_true_s, y_pred_s,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

# Summary table
summary_df = pd.DataFrame(per_stock_rows).sort_values("accuracy", ascending=False)
print("── Per-Stock Accuracy Summary ──")
print(summary_df.to_string(index=False))

# Bar chart: per-stock accuracy
fig, ax = plt.subplots(figsize=(max(6, len(stock_tickers) * 0.6 + 2), 4))
ax.bar(summary_df["stock"], summary_df["accuracy"], color="steelblue", edgecolor="white")
ax.axhline(1 / n_classes, color="red", linestyle="--", label=f"Random baseline ({1/n_classes:.2f})")
ax.set_xlabel("Stock")
ax.set_ylabel("Accuracy")
ax.set_title("Per-Stock Test Accuracy")
ax.set_ylim(0, 1)
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("per_stock_accuracy.png", dpi=150)
plt.close()
print("Saved → per_stock_accuracy.png")

# Per-stock confusion matrices (one subplot per stock)
n_stocks = len(stock_tickers)
ncols    = min(n_stocks, 4)
nrows    = (n_stocks + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
axes_flat = np.array(axes).flatten() if n_stocks > 1 else [axes]

for idx, ticker in enumerate(stock_tickers):
    mask     = test_stocks == ticker
    cm_s     = confusion_matrix(y_test[mask], y_pred[mask], labels=[0, 1, 2])
    sns.heatmap(
        cm_s, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes_flat[idx],
    )
    axes_flat[idx].set_title(ticker)
    axes_flat[idx].set_xlabel("Predicted")
    axes_flat[idx].set_ylabel("Actual")

# Hide any unused subplots
for idx in range(n_stocks, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle("Per-Stock Confusion Matrices — Test Set", fontsize=14)
plt.tight_layout()
plt.savefig("per_stock_confusion.png", dpi=150)
plt.close()
print("Saved → per_stock_confusion.png")

# ──────────────────────────────────────────────
# 11. Confusion matrix plot (overall)
# ──────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Overall) — Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved → confusion_matrix.png")

# ──────────────────────────────────────────────
# 12. Training curves plot
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Accuracy
axes[1].plot(history.history["accuracy"],     label="Train Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.suptitle("LSTM Training Curves", fontsize=14)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()
print("Saved → training_curves.png")
print("\nDone. Best model saved → model.keras")
print("Outputs: model.keras, confusion_matrix.png, per_stock_accuracy.png, per_stock_confusion.png, training_curves.png")
