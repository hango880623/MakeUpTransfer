import pandas as pd

if __name__ == '__main__':
    label_path = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Pixel/0116/lips_merge/label.csv'
    df = pd.read_csv(label_path)
    print("Column names:")
    print(df.columns.tolist())
    
    # Generate lists of target filenames
    ranges = [55, 60, 65, 70, 80]
    all_targets = [['p_{}_{}_5700_pixel.jpg'.format(i, r) for r in ranges] for i in range(4)]
    
    # Print each target list
    for target in all_targets:
        print(target)
        
        # Filter the DataFrame to only include rows where 'Filename' is in the target list
        filtered_df = df[df['Filename'].isin(target)]
        
        # Get the list of values from the 'Lips_Color' column
        lips_color_list = filtered_df['Lips_Color'].tolist()
        print("Lips_Color list:")
        print(lips_color_list)
