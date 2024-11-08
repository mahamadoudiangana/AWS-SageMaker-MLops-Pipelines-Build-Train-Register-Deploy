
import joblib
import os

"""
In my case, I don't require to have predict_fn(). 
This predict_fn is used to change the prediction. Eg: Adding type or adding more layers to my prediction
"""




def input_fn(request_body, request_content_type):
    print(request_body)
    print(request_content_type)
    if request_content_type == "text/csv":  # If the incomming request body data is a 'text/csv' type
        request_body = request_body.strip() # We will make sure that there is no leading or ending space character in the request body and making sure that the request data is clean 
                                            # and ready for being transformed into a pandas dataframe 
        try:
            df = pd.read_csv(StringIO(request_body), header=None) # Since the the incomming request will come as a stream data
            return df
        
        except Exception as e:
            print(e)
    else:
        return """Please use Content-Type = 'text/csv' and, send the request!!""" 




# The model_fn function is required
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model



def predict_fn(input_data, model):
    if type(input_data) != str: # Because we have transformed the received request body data into a pandas dataframe, so we will only predict if the request body data is a dataframe and not a string
        prediction = model.predict(input_data)
        print(prediction)
        return prediction
    else:
        return input_data
        
        

def output_fn(prediction, content_type):
    import json
    if content_type == 'text/csv': # In our case, we want the response from the model to be a list of numbers with the lenght of the incoming request data points lenght(text/csv) and not a 'application/json':
        response_body = json.dumps(prediction.tolist())
        return response_body
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))
