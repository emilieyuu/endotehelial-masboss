from datetime import datetime

def save_df_to_csv(df, directory, base_name):
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{base_name}.csv"

    path = directory / file_name
    stem = path.stem
    suffix = path.suffix

    df.to_csv(path, index=False)
    print(f"DEBUG: File {path.name} successfully written to directory: {directory}")


def generate_ko_model(base_model, mutation):
        m = base_model.copy()

        for node, state in mutation.items():
            m.mutate(node, state)

        return m