import pandas as pd

def process_wine_data():
    # 1. Read Raw Data
    try:
        df = pd.read_csv('winequality-white.csv', sep=';')
        print("Raw data read successfully. Number of Samples: {}, Number of Characteristics: {}".format(df.shape[0], df.shape[1]))
    except FileNotFoundError:
        print("Error: File Not Found")
        return

    # ----- 1. Create Binary Classification Dataset -----
    df_binary = df.copy()
    
    # New label Column
    df_binary['High_Quality'] = (df_binary['quality'] >= 7).astype(int)
    
    # Delete Original "quality" Column
    df_binary = df_binary.drop(columns=['quality'])
    
    # Make sure the column "High_Quality" located at the last column 
    cols_binary = [c for c in df_binary.columns if c != 'High_Quality'] + ['High_Quality']
    df_binary = df_binary[cols_binary]
    
    # Save
    df_binary.to_csv('wine_binary.csv', index=False)
    print(f"\n Binary Classification Dataset saved as 'wine_binary.csv'")
    print(f"Label Distribution:\n{df_binary['High_Quality'].value_counts().sort_index()}")

    # ----- 2. Create Multi-class Classification -----
    df_multi = df.copy()
    
    # change original "quality" to "Quality_Tier"
    def categorize_quality(score):
        if score <= 5:
            return 0 # Low
        elif score == 6:
            return 1 # Medium
        else:
            return 2 # High
            
    df_multi['Quality_Tier'] = df_multi['quality'].apply(categorize_quality)
    
    # Delete Original "quality" Column
    df_multi = df_multi.drop(columns=['quality'])
    
    # Make sure the column "Quality_Tier" located at the last column 
    cols_multi = [c for c in df_multi.columns if c != 'Quality_Tier'] + ['Quality_Tier']
    df_multi = df_multi[cols_multi]
    
    # Save
    df_multi.to_csv('wine_multiclass.csv', index=False)
    print(f"\n Multi-class Classification Dataset saved as 'wine_multiclass.csv'")
    print(f"Label Distribution:\n{df_multi['Quality_Tier'].value_counts().sort_index()}")

if __name__ == "__main__":
    process_wine_data()
