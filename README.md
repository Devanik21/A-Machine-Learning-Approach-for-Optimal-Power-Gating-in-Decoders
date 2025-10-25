# ⚡ Power Gating Decoder ML Optimizer

Advanced Machine Learning System for Low-Power Decoder Design Optimization

## 📋 Setup Instructions

### 1. GitHub Repository Setup

1. Create a new GitHub repository (or use existing one)
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `decoder_power_delay_area_dataset.csv`

### 2. Update Dataset Path in app.py

In `app.py`, find line ~65 and update with your GitHub username and repo name:

```python
url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/decoder_power_delay_area_dataset.csv"
```

**Example:**
```python
url = "https://raw.githubusercontent.com/john_doe/power-decoder-ml/main/decoder_power_delay_area_dataset.csv"
```

### 3. Deploy on Streamlit Cloud

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `app.py`
6. Click "Deploy"!

### 4. Local Testing (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 📊 Dataset Format

Your `decoder_power_delay_area_dataset.csv` must have these columns:

| Column | Description | Unit |
|--------|-------------|------|
| decoder_size | Decoder bits (2-6) | bits |
| tech_node | Technology node | nm |
| supply_voltage | Supply voltage | V |
| threshold_voltage | Threshold voltage | V |
| transistor_width | Transistor width | µm |
| load_capacitance | Load capacitance | fF |
| pg_efficiency | Power gating efficiency | 0-1 |
| switching_activity | Switching activity | 0-1 |
| leakage_factor | Leakage factor | 0-1 |
| temperature | Operating temperature | °C |
| **power** | Power consumption (target) | mW |
| **delay** | Propagation delay (target) | ns |
| **area** | Silicon area (target) | µm² |

## 🚀 Features

- 🤖 **4 ML Algorithms**: Random Forest, Gradient Boosting, Neural Networks, SVR
- 📊 **Interactive Dashboard**: Real-time predictions and visualizations
- 🎯 **Multi-Objective Optimization**: Power-Delay-Area trade-off analysis
- 📈 **3D Pareto Front**: Beautiful interactive visualization
- 📄 **Auto-Generated Reports**: Conference paper ready documentation
- 💾 **Export Functionality**: Download results and reports

## 🎯 Usage

1. **Data Overview Tab**: Explore your dataset statistics and distributions
2. **ML Training Tab**: Train models and compare performance
3. **Predictions Tab**: Make real-time predictions for custom configurations
4. **Optimization Tab**: Find Pareto-optimal designs
5. **Report Tab**: Generate comprehensive project report

## 🛠️ Technologies

- **Framework**: Streamlit
- **ML Library**: scikit-learn
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## 📝 Project Details

**Project:** Power Gating in Decoders - Low Power Strategy  
**Course:** B.Tech Final Year Project (045-320)  
**Topic:** ML-Driven Multi-Objective Optimization

## 🎓 For Conference Paper

The app automatically generates:
- Comprehensive methodology section
- Results and comparisons
- Feature importance analysis
- Pareto-optimal configurations
- Complete project report

## 💡 Tips

- Use the sidebar to configure ML algorithms and optimization targets
- Export datasets and reports for your documentation
- The 3D Pareto front is perfect for presentations!
- Use the generated report for your conference paper

## 📞 Support

For issues or questions:
1. Check that dataset path in app.py is correct
2. Verify CSV file has all required columns
3. Ensure GitHub repository is public

---

**Good luck with your project! 🚀**
