import pandas as pd
import numpy as np

def run_comprehensive_analysis(csv_path: str = "evaluation_results.csv"):
    """
    Perform comprehensive analysis on the evaluation results:
    1. Create abstention rate matrix by retrieval type and prompt
    2. Create abstention rate matrices split by knowledge base
    3. Check abstention/factuality consistency
    4. Create correctness rate matrix by retrieval type and prompt
    5. Create correctness rate matrices split by knowledge base
    """
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV file with {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return
    
    # Define the custom order for retrieval types
    custom_order = ['DIRECT', 'LONG_IN_CONTEXT', 'BASIC_RAG', 'HYDE_RAG']
    
    # Get list of unique knowledge bases
    knowledge_bases = df['knowledge_base_identifier'].unique()
    
    # 1. OVERALL ABSTENTION RATE MATRIX
    print("\n" + "="*80)
    print("OVERALL ABSTENTION RATE MATRIX")
    print("="*80)
    
    # Create binary column for abstention count
    df['is_abstained'] = (df['abstention'] == 'yes').astype(int)
    
    # Create pivot table for abstention
    pivot_abstention = pd.pivot_table(
        df,
        index='retrieval_type',
        columns='system_prompt_name',
        values='is_abstained',
        aggfunc='mean',
        fill_value=np.nan
    )
    
    # Reorder rows
    pivot_abstention = pivot_abstention.reindex(custom_order)
    
    # Format for display
    formatted_abstention = format_pivot_as_percentages(pivot_abstention)
    print(formatted_abstention)
    
    # 2. ABSTENTION RATE MATRICES BY KNOWLEDGE BASE
    print("\n" + "="*80)
    print("ABSTENTION RATE MATRICES BY KNOWLEDGE BASE")
    print("="*80)
    
    for kb in knowledge_bases:
        print(f"\nKnowledge Base: {kb}")
        print("-" * 50)
        
        # Filter data for this knowledge base
        kb_df = df[df['knowledge_base_identifier'] == kb]
        
        # Create pivot table
        kb_pivot = pd.pivot_table(
            kb_df,
            index='retrieval_type',
            columns='system_prompt_name',
            values='is_abstained',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        # Reorder rows
        kb_pivot = kb_pivot.reindex(custom_order)
        
        # Format for display
        formatted_kb_pivot = format_pivot_as_percentages(kb_pivot)
        print(formatted_kb_pivot)
    
    # 3. CHECK ABSTENTION/FACTUALITY CONSISTENCY
    print("\n" + "="*80)
    print("ABSTENTION/FACTUALITY CONSISTENCY CHECK")
    print("="*80)
    
    # Print unique values
    print("\nUnique abstention values:")
    print(df['abstention'].unique())
    print("\nUnique factuality values:")
    print(df['factuality'].unique())
    
    # Count NaN values
    print(f"\nRows with NaN factuality: {df['factuality'].isna().sum()}")
    print(f"Rows with non-NaN factuality: {df['factuality'].notna().sum()}")
    
    # Check for inconsistencies
    inconsistent_yes = df[(df['abstention'] == 'yes') & (df['factuality'].notna())]
    inconsistent_no = df[(df['abstention'] == 'no') & (df['factuality'].isna())]
    
    # Print results
    print("\nChecking abstention-factuality consistency:")
    print(f"Total rows: {len(df)}")
    print(f"Rows with abstention='yes': {len(df[df['abstention'] == 'yes'])}")
    print(f"Rows with NaN factuality: {df['factuality'].isna().sum()}")
    
    if len(inconsistent_yes) > 0:
        print(f"\n⚠️ Found {len(inconsistent_yes)} rows where abstention='yes' but factuality is not NaN")
    else:
        print("\n✓ All rows with abstention='yes' have NaN factuality")
    
    if len(inconsistent_no) > 0:
        print(f"\n⚠️ Found {len(inconsistent_no)} rows where abstention='no' but factuality is NaN")
    else:
        print("✓ All rows with abstention='no' have non-NaN factuality")
    
    # Print factuality distribution
    print("\nDistribution of factuality values for abstention='no':")
    print(df[df['abstention'] == 'no']['factuality'].value_counts(dropna=False))
    
    # 4. OVERALL CORRECTNESS RATE MATRIX
    print("\n" + "="*80)
    print("OVERALL CORRECTNESS RATE MATRIX (tier_1 + tier_2)")
    print("="*80)
    
    # Create binary column for correctness
    df['is_correct'] = df['factuality'].isin(['tier_1', 'tier_2']).astype(int)
    
    # Filter to only include rows where abstention is "no"
    df_no_abstention = df[df['abstention'] == 'no']
    
    # Create pivot table for correctness
    pivot_correctness = pd.pivot_table(
        df_no_abstention,
        index='retrieval_type',
        columns='system_prompt_name',
        values='is_correct',
        aggfunc='mean',
        fill_value=np.nan
    )
    
    # Reorder rows
    pivot_correctness = pivot_correctness.reindex(custom_order)
    
    # Format for display
    formatted_correctness = format_pivot_as_percentages(pivot_correctness)
    print(formatted_correctness)
    
    # 5. CORRECTNESS RATE MATRICES BY KNOWLEDGE BASE
    print("\n" + "="*80)
    print("CORRECTNESS RATE MATRICES BY KNOWLEDGE BASE (tier_1 + tier_2)")
    print("="*80)
    
    for kb in knowledge_bases:
        print(f"\nKnowledge Base: {kb}")
        print("-" * 50)
        
        # Filter data for this knowledge base
        kb_df_no_abstention = df_no_abstention[df_no_abstention['knowledge_base_identifier'] == kb]
        
        # Create pivot table
        kb_pivot = pd.pivot_table(
            kb_df_no_abstention,
            index='retrieval_type',
            columns='system_prompt_name',
            values='is_correct',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        # Reorder rows
        kb_pivot = kb_pivot.reindex(custom_order)
        
        # Format for display
        formatted_kb_pivot = format_pivot_as_percentages(kb_pivot)
        print(formatted_kb_pivot)

def format_pivot_as_percentages(pivot):
    """Format a pivot table with percentages for display"""
    formatted = pd.DataFrame(
        index=pivot.index,
        columns=pivot.columns
    )
    
    for col in pivot.columns:
        for idx in pivot.index:
            if idx in pivot.index and col in pivot.columns:
                if not pd.isna(pivot.loc[idx, col]):
                    formatted.loc[idx, col] = f"{pivot.loc[idx, col]:.2%}"
                else:
                    formatted.loc[idx, col] = "NA"
    
    return formatted

if __name__ == "__main__":
    run_comprehensive_analysis()