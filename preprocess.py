import pandas as pd
import re
import os
import sys

def get_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
    except ImportError:
        return "Unknown (install psutil for memory tracking)"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    # Remove special characters and emojis (basic cleaning)
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase and strip
    return text.lower().strip()

def preprocess_data(input_file, output_file):
    print(f"Starting preprocessing of {input_file}...")
    print(f"Current Memory Usage: {get_memory_usage()}")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load necessary columns
    cols = ['tweet_id', 'text', 'inbound', 'in_response_to_tweet_id']
    try:
        print("Loading dataset (this may take a moment)...")
        # Use a slightly more memory-efficient dtype for tweet_ids if possible
        # But for now, we'll stick to default and monitor
        df = pd.read_csv(input_file, usecols=cols, low_memory=False)
        print(f"Dataset loaded. Memory Usage: {get_memory_usage()}")
        
        print("Filtering company responses...")
        responses = df[(df['inbound'] == False) & (df['in_response_to_tweet_id'].notna())].copy()
        print(f"Responses filtered. Memory Usage: {get_memory_usage()}")
        
        print("Merging responses with original queries...")
        # Ensure IDs are comparable
        responses['in_response_to_tweet_id'] = responses['in_response_to_tweet_id'].astype(float)
        df['tweet_id'] = df['tweet_id'].astype(float)
        
        merged = responses.merge(
            df[['tweet_id', 'text']], 
            left_on='in_response_to_tweet_id', 
            right_on='tweet_id', 
            suffixes=('_resp', '_query')
        )
        print(f"Merged {len(merged)} pairs. Memory Usage: {get_memory_usage()}")
        
        # Free up some memory
        del df
        del responses
        
        print("Cleaning text...")
        merged['query'] = merged['text_query'].apply(clean_text)
        merged['response'] = merged['text_resp'].apply(clean_text)
        
        # Filter out empty pairs
        final_df = merged[(merged['query'] != "") & (merged['response'] != "")][['query', 'response']]
        print(f"Pairs cleaned. Final pair count: {len(final_df)}. Memory Usage: {get_memory_usage()}")
        
        print(f"Saving to {output_file}...")
        final_df.to_csv(output_file, index=False)
        print("Preprocessing complete.")
        
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    preprocess_data('twcs.csv', 'processed_qa.csv')
