from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url = "https://detect.roboflow.com",
    api_key="fJh6XuKHn8TCVn4jLU0w"
)

result = client.run_workflow(
    workspace_name="specializationproject",
    workflow_id="detect-count-and-visualize",
    images={
        "image": "03.png"
    },
    use_cache=True # cache workflow definition for 15 minutes
)

# --- MODIFIED PART STARTS HERE ---

# The result from a workflow is a list, so we access the first item
workflow_output = result[0]

# Access the list of predictions from the output
# The structure is result -> 'predictions' dictionary -> 'predictions' list
predictions = workflow_output['predictions']['predictions'] # 

print(f"Found {len(predictions)} object(s):")

# Loop through each prediction found in the image
for prediction in predictions:
    # Get the class name and confidence score for the current prediction
    class_name = prediction['class'] # 
    confidence = prediction['confidence'] # 
    
    # Print the extracted information in a readable format
    print(f" - Class: {class_name}, Confidence: {confidence:.2%}")
