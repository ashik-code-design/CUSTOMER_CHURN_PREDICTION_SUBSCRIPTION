"""Build temporal customer features from dated monthly records.

Input is a long table with customer_id, event_month, logins_30d,
usage_hours_30d, support_tickets_30d, payment_delay_days.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {"customer_id","event_month","logins_30d","usage_hours_30d",
            "support_tickets_30d","payment_delay_days"}

def build_temporal_features(history: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED-set(history.columns))
    if missing:
        raise ValueError("Missing monthly-history fields: "+", ".join(missing))
    d=history.copy()
    d["event_month"]=pd.to_datetime(d["event_month"],errors="coerce")
    if d["event_month"].isna().any():
        raise ValueError("event_month contains invalid dates.")
    d=d.sort_values(["customer_id","event_month"])
    for col in ["logins_30d","usage_hours_30d","support_tickets_30d","payment_delay_days"]:
        d[col]=pd.to_numeric(d[col],errors="coerce").fillna(0)
    result=[]
    for cid,g in d.groupby("customer_id"):
        g=g.sort_values("event_month")
        recent=g.tail(3)
        prior=g.iloc[max(0,len(g)-6):max(0,len(g)-3)]
        last=recent.iloc[-1]
        def trend(col):
            if len(prior)==0: return np.nan
            a=prior[col].mean()
            b=recent[col].mean()
            return float((b-a)/(abs(a)+1e-6))
        result.append({
            "customer_id":str(cid),
            "history_months_available":int(len(g)),
            "login_trend_3m_vs_prior3m":trend("logins_30d"),
            "usage_trend_3m_vs_prior3m":trend("usage_hours_30d"),
            "support_ticket_trend_3m_vs_prior3m":trend("support_tickets_30d"),
            "payment_delay_trend_3m_vs_prior3m":trend("payment_delay_days"),
            "recent_avg_logins_3m":float(recent.logins_30d.mean()),
            "recent_avg_usage_hours_3m":float(recent.usage_hours_30d.mean()),
            "recent_avg_support_tickets_3m":float(recent.support_tickets_30d.mean()),
            "recent_avg_payment_delay_days_3m":float(recent.payment_delay_days.mean()),
            "last_event_month":last.event_month.strftime("%Y-%m-%d"),
            "data_provenance":str(g.data_provenance.iloc[-1]) if "data_provenance" in g else "SOURCE_DATA",
        })
    return pd.DataFrame(result)

def main():
    inp=ROOT/"data"/"customer_monthly_history_demo.csv"
    out=ROOT/"data"/"temporal_features_demo.csv"
    history=pd.read_csv(inp)
    features=build_temporal_features(history)
    features.to_csv(out,index=False)
    print(f"Wrote {len(features)} customer temporal feature rows to {out}")
    if "data_provenance" in history and history.data_provenance.eq("SYNTHETIC_DEMO_ONLY").all():
        print("NOTICE: these trends are generated from synthetic demo history, not observed customer events.")
if __name__=="__main__": main()
