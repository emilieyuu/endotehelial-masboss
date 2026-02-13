
def save_df_to_csv(df, directory, file_name):
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / file_name
    stem = path.stem
    suffix = path.suffix

    counter = 1
    while path.exists():
        new_name = f"{stem}_{counter:02d}{suffix}"
        path = directory / new_name
        counter += 1

    df.to_csv(path, index=False)
    print(f"DEBUG: File {path.name} successfully written to directory: {directory}")


def generate_ko_model(base_model, mutation):
        m = base_model.copy()

        for node, state in mutation.items():
            m.mutate(node, state)

        return m