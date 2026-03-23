import pandas as pd

#def find_overlaps(excel_file, sheet_name='Sheet1', col1_name=None, col2_name=None, col1_idx=0, col2_idx=1):
def find_overlaps(harmful, harmless):
    """
    Find overlapping elements between two columns in an Excel file.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file
    sheet_name : str, default='Sheet1'
        Name of the sheet containing the data
    col1_name, col2_name : str, optional
        Names of the columns to compare. If not provided, will use column indices.
    col1_idx, col2_idx : int, default=0,1
        Indices of the columns to compare (0-based) if names aren't provided
        
    Returns:
    --------
    list
        List of overlapping elements
    """
    # Read the Excel file
    df1 = pd.read_csv(harmful)
    df2 = pd.read_csv(harmless)
    # Determine which columns to use (by name or index)

    col1 = df1.iloc[0].dropna().tolist()
    col2 = df2.iloc[0].dropna().tolist()
    
    # Find overlapping elements (maintaining order of first column)
    overlaps = [item for item in col1 if item in col2]
    
    return overlaps

if __name__ == "__main__":
    # Example usage
    harmful = '/Users/lilucen/Desktop/research/Intervention/codebase/YouTube-Sock-Puppet/data/training/harmful.csv'
    harmless = '/Users/lilucen/Desktop/research/Intervention/codebase/YouTube-Sock-Puppet/data/training/non_harmful.csv' 
    # Option 1: Using column indices (first and second columns)
    overlaps = find_overlaps(harmful, harmless)
    
    # Option 2: Using column names
    # overlaps = find_overlaps(file_path, col1_name="Column1", col2_name="Column2")
    
    print("Overlapping elements:")
    for item in overlaps:
        print(f"- {item}")
    
    print(f"\nTotal overlaps found: {len(overlaps)}")
    
    # # Optionally save results to a new Excel file
    # if overlaps:
    #     result_df = pd.DataFrame({"videoID": overlaps})
    #     result_df.to_csv("harmful_overlaps_results.csv", index=False)
    #     print("\nResults saved to 'harmful_overlaps_results.csv'")