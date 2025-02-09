import os
import subprocess
import sys

# Create a virtual environment
def create_environment(env_name):
    print(f"Creating virtual environment: {env_name}")
    subprocess.check_call([sys.executable, "-m", "venv", env_name])

# Install dependencies from a requirements file
def install_dependencies(env_name, requirements_file):
    print("Installing dependencies...")
    pip_executable = os.path.join(env_name, "Scripts", "pip") if os.name == "nt" else os.path.join(env_name, "bin", "pip")
    subprocess.check_call([pip_executable, "install", "-r", requirements_file])

if __name__ == "__main__":
    # Name of the virtual environment
    env_name = "venv"
    # Path to the requirements file
    requirements_file = "requirements.txt"

    try:
        # Create a virtual environment
        create_environment(env_name)

        # Install dependencies from the requirements file
        if os.path.exists(requirements_file):
            install_dependencies(env_name, requirements_file)
        else:
            print(f"Requirements file '{requirements_file}' not found.")

        print("Environment setup is complete. You can now work within this environment.")

    except Exception as e:
        print(f"An error occurred: {e}")