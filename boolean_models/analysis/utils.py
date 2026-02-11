
def save_df_to_csv(df, directory, file_name): 
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / file_name
    df.to_csv(path, index=False)
    print(f"DEBUG: File {file_name} successfully written to directory: {directory}")