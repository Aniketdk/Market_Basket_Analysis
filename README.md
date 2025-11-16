# Market_Basket_Analysis

## What's included

- `market_basket_demo_20k.csv` — Demo transactional dataset (20,000+ transactions). **Located at the root of the package.**
- `market_basket_analysis.py` — Python script to build the basket, run Apriori (using `mlxtend`) and save association rules.
- `POWER_BI_BUILD_INSTRUCTIONS.md` — Step-by-step Power BI guide to build the 3-page dashboard described in the project.
- `requirements.txt` — Python dependencies to run the notebook/script.

## How to run (local machine)

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python market_basket_analysis.py
```

This will produce `retail_data.csv` in the package folder.
