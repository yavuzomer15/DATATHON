import pandas as pd
import numpy as np

EVENT_VIEW, EVENT_CART, EVENT_REMOVE, EVENT_BUY = "VIEW","ADD_CART","REMOVE_CART","BUY"

def build_session_features(df, with_target=False):
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.sort_values(["user_session","event_time"], kind="mergesort")

    # Event counts
    counts = (df.groupby(["user_session","event_type"])
                .size().unstack(fill_value=0).reset_index())
    cols = [c for c in counts.columns if c!="user_session"]
    counts["total_events"] = counts[cols].sum(axis=1)

    # Diversity
    u_prod = df.groupby("user_session")["product_id"].nunique().reset_index(name="unique_products")
    u_cat  = df.groupby("user_session")["category_id"].nunique().reset_index(name="unique_categories")

    # Time
    span = df.groupby("user_session")["event_time"].agg(["min","max"]).reset_index()
    span["session_duration"]   = (span["max"]-span["min"]).dt.total_seconds()
    span = span[["user_session","session_duration"]]
    avg_int = df.groupby("user_session")["event_time"].apply(
        lambda x: (x.max()-x.min()).total_seconds()/(len(x)-1) if len(x)>1 else 0
    ).reset_index(name="avg_event_interval")

    # Merge
    feat = (counts.merge(u_prod, on="user_session", how="left")
                  .merge(u_cat,  on="user_session", how="left")
                  .merge(span,   on="user_session", how="left")
                  .merge(avg_int,on="user_session", how="left"))

    # Ratios
    feat["cart_per_view"] = feat.get(EVENT_CART,0) / (feat.get(EVENT_VIEW,0)+1)
    feat["buy_per_cart"]  = feat.get(EVENT_BUY,0)  / (feat.get(EVENT_CART,0)+1)
    feat["buy_per_view"]  = feat.get(EVENT_BUY,0)  / (feat.get(EVENT_VIEW,0)+1)

    # Last event & flags
    last_ev = df.groupby("user_session")["event_type"].last().reset_index(name="last_event_type")
    feat = feat.merge(last_ev, on="user_session", how="left")
    feat["has_buy"] = (feat.get(EVENT_BUY,0) > 0).astype(int)

    # Sequence patterns
    def has_pattern(evts, pattern):
        i=0
        for e in evts:
            if e==pattern[i]:
                i+=1
                if i==len(pattern): return 1
        return 0

    rows=[]
    for sid,g in df.groupby("user_session", sort=False):
        evts = g["event_type"].tolist()
        rows.append({
            "user_session": sid,
            "pattern_view_cart":      has_pattern(evts, [EVENT_VIEW, EVENT_CART]),
            "pattern_cart_buy":       has_pattern(evts, [EVENT_CART, EVENT_BUY]),
            "pattern_view_cart_buy":  has_pattern(evts, [EVENT_VIEW, EVENT_CART, EVENT_BUY]),
        })
    patterns = pd.DataFrame(rows)
    feat = feat.merge(patterns, on="user_session", how="left")

    # Target ve grup (user_id) – CV için lazım
    # (Her session tek kullanıcıya ait olduğu için mode/first güvenli)
    user_map = df.groupby("user_session")["user_id"].first().reset_index()
    feat = feat.merge(user_map, on="user_session", how="left")

    if with_target:
        target = df[["user_session","session_value"]].drop_duplicates()
        feat = feat.merge(target, on="user_session", how="left")

    # Nümerikleri doldur
    num_cols = feat.select_dtypes(include=[np.number]).columns
    feat[num_cols] = feat[num_cols].fillna(0)

    return feat

# Çalıştır
train = pd.read_csv("C:/Users/omery/Desktop/python/DATATHON/train.csv")
test  = pd.read_csv("C:/Users/omery/Desktop/python/DATATHON/test.csv")

train_features = build_session_features(train, with_target=True)
test_features  = build_session_features(test,  with_target=False)

train_features.to_csv("train_features.csv", index=False)
test_features.to_csv("test_features.csv", index=False)
