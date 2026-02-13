
def save_df_to_csv(df, directory, file_name): 
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / file_name
    df.to_csv(path, index=False)
    print(f"DEBUG: File {file_name} successfully written to directory: {directory}")

def generate_ko_model(base_model, mutation):
        m = base_model.copy()

        for node, state in mutation.items():
            m.mutate(node, state)

        return m