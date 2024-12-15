## Files Overview
- **`main.py`**: The main entry point for running the simulation.
- **`env.py`**: Defines the simulation environments.
- **`policy.py`**: Placeholder for any basic policies (for testing purposes).
- **`requirements.txt`**: Lists all required Python dependencies.
- **`Dockerfile`**: Used to containerize the application for easier setup and execution.
- **`test.py`**: Scripts for testing functionality.

---

## How to Run the Project

1. Ensure **Docker** is installed on your system.
2. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/sanwalhussain/try.git
   cd try

3. Ensure **Docker** is installed on your system.
   ```bash
      docker build -t metadrive-app .

4. Ensure **Docker** is installed on your system.
   ```bash
   docker run -d --name metadrive-container -p 8080:8080 metadrive-app

5. Ensure **Docker** is installed on your system.
   ```bash
      docker logs -f metadrive-container


**Model Saving:** The trained models will be automatically saved in the model_path directory within the project.


