# pseudo-structure for inference.py

def model_fn(model_dir):
    # load xgb_fraud_model.pkl + model_metadata.json from model_dir
    return my_custom_model_object

def input_fn(request_body, request_content_type):
    # parse JSON → dict or DataFrame
    return parsed_input

def predict_fn(input_object, model):
    # call build_features_for_inference + predict_transaction_with_decision
    return prediction_dict

def output_fn(prediction, accept):
    # return JSON string
    return json.dumps(prediction)
