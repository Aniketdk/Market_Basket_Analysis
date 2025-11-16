import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# -------------------------------
# STEP 1 — Load dataset
# -------------------------------

def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows.")
    return df


# -------------------------------
# STEP 2 — Convert to transaction format
# -------------------------------

def convert_to_transactions(df, col_order="OrderID", col_item="Item"):
    print("Transforming dataset into transaction format...")

    # Group items by Order ID
    transactions = (
        df.groupby(col_order)[col_item]
        .apply(list)
        .values.tolist()
    )

    print(f"Created {len(transactions)} transactions.")
    return transactions


# -------------------------------
# STEP 3 — Encode transactions
# -------------------------------

def encode_transactions(transactions):
    print("Encoding transactions...")

    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_data, columns=te.columns_)

    print(f"Encoded data shape: {df_encoded.shape}")
    return df_encoded


# -------------------------------
# STEP 4 — Run Apriori
# -------------------------------

def run_apriori(df_encoded, min_support=0.01):
    print("Running Apriori algorithm...")

    frequent_items = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True
    )

    print(f"Found {len(frequent_items)} frequent itemsets.")
    return frequent_items


# -------------------------------
# STEP 5 — Generate association rules
# -------------------------------

def generate_rules(frequent_items, min_confidence=0.3, min_lift=1.0):
    print("Generating association rules...")

    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=min_confidence
    )

    rules = rules[rules["lift"] >= min_lift]

    print(f"Generated {len(rules)} rules.")
    return rules


# -------------------------------
# STEP 6 — Save output
# -------------------------------

def save_output(rules, output_path):
    rules.to_csv(output_path, index=False)
    print(f"Rules exported to {output_path}")


# -------------------------------
# MAIN EXECUTION
# -------------------------------

if __name__ == "__main__":
    INPUT_FILE = "market_basket_demo_20k.csv"
    OUTPUT_FILE = "market_basket_rules_output.csv"

    # 1. Load dataset
    df = load_data(INPUT_FILE)

    # 2. Convert to transactions
    transactions = convert_to_transactions(df)

    # 3. Encode
    df_encoded = encode_transactions(transactions)

    # 4. Apriori
    frequent_items = run_apriori(df_encoded, min_support=0.01)

    # 5. Rules
    rules = generate_rules(
        frequent_items,
        min_confidence=0.3,
        min_lift=1.1
    )

    # 6. Save results
    save_output(rules, OUTPUT_FILE)

    print("\nMarket Basket Analysis Completed Successfully!")
