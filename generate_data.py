import pandas as pd
import numpy as np

# Configuration
OUTPUT_FILE = "tenant_data.csv"
NUM_SAMPLES = 10000

def generate_synthetic_data(n_samples=10000):
    """
    Generates fake tenant data based on general rules with some noise.
    
    General Rule:
    - GOOD (1): missedPeriods <= 3 AND totalDisputes <= 3
    - BAD (0): Otherwise
    - With 10% chance, flip the label for noise
    """
    np.random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # 1. Randomly generate base features
        # Adjusted probabilities so we actually get enough "Bad" users for the model to learn
        missed_periods = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 8, 10, 12], 
            p=[0.30, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05] 
            # Note: Values 6, 8, 10, 12 satisfy the (>5) condition
        )
        
        total_disputes = np.random.choice(
            [0, 1, 2, 3, 4, 5, 8], 
            p=[0.60, 0.10, 0.05, 0.05, 0.05, 0.10, 0.05]
            # Note: Values 4, 5, 8 satisfy the (>3) condition
        )
        
        # Days since last dispute (Removed as per user request)
        
        # 2. LOGIC TO ASSIGN LABELS (The "Truth")
        # General Rule: Good if both <=3, Bad otherwise
        if missed_periods <= 3 and total_disputes <= 3:
            label = 1 # Good / Trustworthy
        else:
            label = 0 # Bad / Risky
        
        # Add noise: 10% chance to flip label
        if np.random.rand() < 0.1:
            label = 1 - label
        
        data.append([missed_periods, total_disputes, label])

    columns = ["missedPeriods", "totalDisputes", "label"]
    df = pd.DataFrame(data, columns=columns)
    
    return df

if __name__ == "__main__":
    print(f"Generating {NUM_SAMPLES} synthetic records...")
    df = generate_synthetic_data(NUM_SAMPLES)
    
    # Check balance to ensure we have enough "Bad" examples
    counts = df['label'].value_counts()
    print(f"Class Distribution:\n{counts}")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Data generated and saved to '{OUTPUT_FILE}'")
    
    # Show a preview
    print("\nPreview:")
    print(df.head(10))