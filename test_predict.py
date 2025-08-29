import pandas as pd, numpy as np, joblib, json

# 1) Artefaktları yükle
with open("meta.json") as f:
    meta = json.load(f)
use_log = meta["use_log"]
X_columns = meta["X_columns"]

models = joblib.load("models_log.pkl") if use_log else joblib.load("models_raw.pkl")
ohe = joblib.load("ohe.pkl")

# 2) Test feature’larını hazırla (train ile birebir aynı)
test_features = pd.read_csv("test_features.csv")

cat_test = ohe.transform(test_features[["last_event_type"]])
cat_cols = [f"last_event_type_{c}" for c in ohe.categories_[0]]
cat_test_df = pd.DataFrame(cat_test, columns=cat_cols, index=test_features.index)

X_test = pd.concat([
    test_features.drop(columns=["user_session","last_event_type","user_id"]),
    cat_test_df
], axis=1)

# Eğitimdeki kolon sırası ile hizala (eksik varsa 0 ile doldur)
X_test = X_test.reindex(columns=X_columns, fill_value=0)

# 3) Ansambl tahmin
def predict_ensemble(models, X):
    return np.column_stack([m.predict(X) for m in models]).mean(axis=1)

test_pred = predict_ensemble(models, X_test)
if use_log:
    test_pred = np.expm1(test_pred)
test_pred = np.clip(test_pred, 0, None)

# 4) sample_submission’a hizala ve kaydet
sub = pd.read_csv("C:/Users/omery/Desktop/python/DATATHON/sample_submission.csv")  # sende farklı path ise onu kullan
pred_df = pd.DataFrame({
    "user_session": test_features["user_session"],
    "session_value": test_pred
})
final_sub = sub[["user_session"]].merge(pred_df, on="user_session", how="left")
final_sub.to_csv("submission_baseline.csv", index=False)
print(final_sub.head(), final_sub.shape)