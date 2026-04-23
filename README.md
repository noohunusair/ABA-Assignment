# 🚇 Delhi Metro Analytics Dashboard

**MBA Applied Business Analytics Project**  
Interactive Streamlit dashboard analysing 150,000+ Delhi Metro trips (2022–2024).

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview Dashboard | KPIs, monthly volume, ticket split, yearly revenue |
| 🚉 Station Intelligence | Top stations, revenue, day-of-week heatmap |
| 🗺️ Route & Distance Analysis | Popular routes, fare vs distance, distance bands |
| 🎫 Ticket & Demand Patterns | Ticket type performance, peak/off-peak breakdown |
| 📅 Time Series Trends | Daily, monthly, YoY, quarterly trends |
| 💡 Insights & Recommendations | Data-driven findings for DMRC & urban planners |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/delhi-metro-dashboard.git
cd delhi-metro-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset in the same folder
#    File: delhi_metro_updated.csv

# 4. Run
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to **GitHub** (make sure `delhi_metro_updated.csv` is included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy** — live in ~2 minutes ✅

---

## 📁 File Structure

```
delhi-metro-dashboard/
│
├── app.py                    ← Main Streamlit dashboard
├── delhi_metro_updated.csv   ← Dataset (150,000 trips)
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   └── config.toml           ← Dark theme config
└── README.md
```

---

## 📦 Tech Stack

- **Streamlit** — Web app framework
- **Pandas** — Data manipulation & cleaning
- **Plotly** — Interactive charts
- **NumPy** — Numerical operations

---

## 📌 Dataset Description

| Column | Description |
|--------|-------------|
| TripID | Unique trip identifier |
| Date | Trip date (2022–2024) |
| From_Station | Origin station |
| To_Station | Destination station |
| Distance_km | Trip distance in km |
| Fare | Ticket fare (₹) |
| Cost_per_passenger | Fare per passenger (₹) |
| Passengers | Number of passengers |
| Ticket_Type | Smart Card / Tourist Card / Single / Return |
| Remarks | Trip condition (peak / off-peak / weekend / festival / maintenance) |

---

*Dashboard designed to mirror professional analytics standards · MBA Applied Business Analytics*
