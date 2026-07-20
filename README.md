<div align="center">

# AI-Powered Refinery Performance Optimization System
### Predictive Intelligence for Refinery Operations

An end-to-end machine learning system that predicts key refinery performance
indicators — boiler efficiency, CDU throughput, energy consumption, and product
yield — from live process data, with explainable AI insights delivered through
an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-FFCC00)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-3776AB)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-9ACD32)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-6A5ACD)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Overview

Modern refineries generate massive volumes of operational data from sensors,
Distributed Control Systems (DCS), and process equipment. Despite continuous
monitoring, engineers often rely on manual analysis and reactive
decision-making — making it difficult to catch performance degradation before
it affects production.

**AI-Powered Refinery Performance Optimization** addresses this by using
machine learning to predict performance indicators directly from process
parameters, helping refinery engineers improve efficiency, reduce energy
consumption, optimize throughput, and maximize product yield — through a
single, explainable, interactive dashboard.

Built for the **IOCL Refinery ML Project**, by a six-member team, delivered
across four sprint phases.

---

## Features

| Feature | Description |
|---|---|
| Boiler Efficiency Prediction | Predicts `Boiler_Eff_pct` from furnace and process parameters to catch efficiency drops early. |
| CDU Throughput Prediction | Forecasts `CDU_Feed_TPH` to support unit-level planning and scheduling. |
| Energy Consumption Prediction | Predicts `Electricity_MWh` (future: `Steam_TPH`, `FuelGas_Nm3h`) to flag abnormal utility usage. |
| Product Yield Prediction | Predicts LPG, Petrol, Diesel, ATF, Naphtha, VGO, and Total Yield (%) from feed and process variables. |
| Explainable AI (SHAP) | Every prediction is paired with SHAP-based feature importance, so engineers know *why*, not just *what*. |
| Interactive Dashboard | A single Streamlit dashboard integrates all four models with live predictions and visual insights. |

---

## Architecture Overview

```
Refinery Sensor / DCS Data
        │
        ▼
┌───────────────────────────┐
│   Data Cleaning + EDA     │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│   Feature Engineering     │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Train-Test Split          │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Benchmark Models           │
│ (CatBoost, XGBoost,        │
│  LightGBM, etc.)           │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Hyperparameter Tuning      │
│ (Optuna)                   │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Final Model + Validation   │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ SHAP Explainability        │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Save Model (Joblib)        │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Streamlit Dashboard        │
│ (Live Predictions + SHAP)  │
└───────────────────────────┘
```

Each of the four modules (Boiler, Throughput, Energy, Yield) follows this
exact pipeline independently, then converges into one shared dashboard.

---

## Tech Stack

**Core & Data Handling**
- Python — Pandas, NumPy, Scikit-Learn

**Modeling**
- CatBoost, XGBoost, LightGBM
- Optuna (hyperparameter tuning)

**Explainability**
- SHAP

**Visualization**
- Matplotlib, Seaborn

**Model Persistence**
- Joblib

**Dashboard**
- Streamlit

**Development & Tooling**
- Jupyter Notebook, VS Code
- Git, GitHub

---

## Dataset Overview

The dataset contains refinery process data across the following categories:

| Category | Example Variables |
|---|---|
| Feed Characteristics | `Crude_API`, `Sulfur_wt_pct` |
| Process Variables | `Furnace_Temp_C`, `Reactor_Temp_C`, `Reflux_Ratio`, `Pressure_Drop_bar` |
| Utility Consumption | `Steam_TPH`, `FuelGas_Nm3h`, `Electricity_MWh`, `CoolingWater_m3h`, `H2_Consumption_Nm3h` |
| Equipment Variables | `Boiler_Eff_pct`, `Fouling_Factor` |
| Production Variables | `CDU_Feed_TPH`, `LPG_Yield_pct`, `Petrol_Yield_pct`, `Diesel_Yield_pct`, `Naphtha_Yield_pct`, `ATF_Yield_pct`, `VGO_Yield_pct`, `Total_Yield_pct` |
| Environmental / Business / Time | `CO2_tpd`, `Refinery_Margin_INR_bbl`, `Timestamp`, `Hour`, `Month` |

---

## Project Scope — Four Core Modules

| Module | Focus | Target Variable(s) |
|---|---|---|
| Module 1 | Boiler Efficiency Prediction | `Boiler_Eff_pct` |
| Module 2 | CDU Throughput Prediction | `CDU_Feed_TPH` |
| Module 3 | Energy Consumption Prediction | `Electricity_MWh` (future: `Steam_TPH`, `FuelGas_Nm3h`) |
| Module 4 | Product Yield Prediction | LPG, Petrol, Diesel, ATF, Naphtha, VGO, Total Yield (%) |

---

## Folder Structure

```
AI-Refinery-Performance-Optimization-System/
├── data/
├── notebooks/
│   ├── Boiler/
│   ├── Throughput/
│   ├── Energy/
│   └── Yield/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── dashboard/
│   └── app.py
├── reports/
├── docs/
├── images/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

```bash
git clone https://github.com/annu2705/AI-Refinery-Performance-Optimization.git
cd AI-Refinery-Performance-Optimization
pip install -r requirements.txt
```

### Train a Model

```bash
python src/train.py
```

### Run a Prediction

```bash
python src/predict.py
```

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard runs at `http://localhost:8501` by default.

---

## Usage

1. **Load Data** — place cleaned process data in `data/`
2. **Train Models** — run each module's pipeline in `notebooks/<Module>/` or via `src/train.py`
3. **View Predictions** — open the Streamlit dashboard and select a module
4. **Explore SHAP Insights** — see which process variables are driving each prediction
5. **Report Issues** — log bugs via the GitHub Issues tab

---

## Team Members

| Member | Role | Responsibilities |
|---|---|---|
| Golden | Project Lead | Architecture, integration, Boiler Efficiency model, GitHub oversight, final dashboard assembly, project management |
| Vidhish | ML Engineer | CDU Throughput model, hyperparameter tuning, SHAP explainability, model validation |
| Ayush | ML Engineer | Energy Consumption model, Product Yield model, SHAP explainability, final model export |
| Yash | Software Engineer | Dashboard, UI design, charts, navigation, alerts, integration support |
| Neha | Documentation Lead | Literature review, project report, architecture diagrams, flowcharts, PPT content, references |
| Annu | GitHub & QA Lead| Repository management, README, testing, screenshots, bug reports, deployment |

---

## Deliverables

**Machine Learning**
- Boiler Efficiency Model · Throughput Model · Energy Model · Product Yield Model

**Software**
- Interactive Streamlit Dashboard

**Documentation**
- Industry-Level Project Report · Technical Documentation · README · Presentation (PPT)

**Visualizations**
- SHAP Plots · Feature Importance Charts · Prediction Graphs · KPI Dashboard

---



## Future Improvements

- Real-time integration with refinery SCADA / IoT systems
- Automated efficiency monitoring with alerts
- Continuous model retraining on new operational data
- Cloud deployment (Docker / Kubernetes)
- CI/CD pipeline for automated versioning and monitoring

---

## Contributing

This is an internal team project. If you're a team member:

1. Create a new branch (`git checkout -b feature/your-feature-name`)
2. Commit your changes (`git commit -m "Add: your feature description"`)
3. Push to the branch (`git push origin feature/your-feature-name`)
4. Open a Pull Request for review before merging into `main`

Please raise updates via the team channel so the `main` branch stays the
single source of truth for the pipeline.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built for the **IOCL Refinery ML Project**

</div>
