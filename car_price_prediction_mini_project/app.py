import sys
import gradio as gr
from src.CarValueML.components.data_transformation import DataTransformationConfig
from src.CarValueML.exception import CustomException
from src.CarValueML.logger import logging
from src.CarValueML.pipelines.predict_pipeline import CustomData, PredictPipeline
from src.CarValueML.utils import load_object

cat_features = ['Fuel', 'Drive', 'Type', 'State', 'Brand', 'Model Name']

# Get the preprocessor object
preprocessor = load_object(DataTransformationConfig().preprocessor_obj_path)
one_hot_feature_arr = preprocessor.named_transformers_["cat_transformer"].get_feature_names_out(cat_features)

# Create a dictionary for the feature and create the list for inputs
def split_and_group(names):
    name_dict = {}
    for name in names:
        parts = name.split("_")
        key = parts[0]
        value = "_".join(parts[1:])
        if key not in name_dict:
            name_dict[key] = []
        name_dict[key].append(value)
    return name_dict

inputs = split_and_group(one_hot_feature_arr)
year_inputs = gr.Number(label="Year", info="In which year the car was bought?")
distance_inputs = gr.Number(label="Distance", info="what is the total distance travelled by the car?")
owner_inputs = gr.Number(label="Owner", info="What are the number of previous owners of the car?")
fuel_inputs = gr.Radio(choices=inputs["Fuel"], label="Fuel", info="Which type of fuel is used in the car?")
drive_inputs = gr.Radio(choices=inputs["Drive"], label="Drive", info="What is the drive type of the car?")
type_inputs = gr.Radio(choices=inputs["Type"], label="Type", info="What is the type of the car?")
state_inputs = gr.Dropdown(choices=inputs["State"], label="State", info="In which state the car was registered?")
brand_inputs = gr.Dropdown(choices=inputs["Brand"], label="Brand", info="What is the brand of the car?")
model_inputs = gr.Dropdown(choices=inputs["Model Name"], label="Model", info="What is the model of the car?")

input_list = [year_inputs, distance_inputs, state_inputs,
              brand_inputs, model_inputs, type_inputs,
              owner_inputs, fuel_inputs, drive_inputs]

def predict_fn(
        year: int, distance: int, state: str, brand: str,
        model: str, type: str, owner: int, fuel: str, drive: str):
    try:
        data = CustomData(year, distance, state, brand,
                          model, type, owner, fuel, drive)
        logging.info("Creating Data as DataFrame")
        data_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        preds = predict_pipeline.predict(data_df)
        logging.info(preds)
        return round(preds)/100
    except Exception as e:
        raise CustomException(e, sys)

demo = gr.Interface(fn=predict_fn,
                    inputs=input_list,
                    outputs=gr.Number(label="Selling Price of the Car", info="in lakhs"),
                    title="Car Price Prediction Application")
demo.launch(share = True)
