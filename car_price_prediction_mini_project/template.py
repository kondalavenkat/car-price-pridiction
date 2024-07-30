import os 
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s: ")
project_name = "CarValueML"

list_of_files = [
    f"data/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/train_pipeline.py",
    f"src/{project_name}/pipelines/predict_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"notebooks/exploratory_data_analysis.ipynb",
    f"notebooks/model_experimentation_ml.ipynb",
    f"templates/index.html",
    f"templates/home.html",
    f"static/style.css",
    f"requirements.txt",
    f"setup.py",
    f"app.py"
]

# Create directories for the files
for filepath in list_of_files:
    filepath = Path(filepath)
    dirname, filename = os.path.split(filepath)
    
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
        logging.info("Creating directory: {}".format(dirname))

    if (not os.path.exists(filepath) or (os.path.getsize(filepath) ==0)):
        with open(filepath, "w") as f:
            pass
        
        logging.info(f"Creating empty files: {filename}")    
        
    else:
        logging.info(f"{filename} already exists")