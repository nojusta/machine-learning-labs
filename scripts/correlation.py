import pandas as pd

def main():
    df = pd.read_csv("./data/clean_data_without_outliers.csv")
    
    correlation_matrix = df.corr(method='pearson')
    
    print("Pearson Correlation Matrix:")
    print(correlation_matrix)

if __name__ == "__main__":
    main()