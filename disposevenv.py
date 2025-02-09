import shutil

# Dispose of a virtual environment
def dispose_environment(env_name):
    print(f"Disposing of virtual environment: {env_name}")
    shutil.rmtree(env_name)

if __name__ == "__main__":
    # Name of the virtual environment
    env_name = "venv"

    # Dispose of the virtual environment
    try:        
        dispose_environment(env_name)
        print("Environment cleanup is complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
