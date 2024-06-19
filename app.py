from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Automatically allow CORS for all routes

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define a route for summarization API
@app.route('/summarize', methods=['POST'])
def summarize():
    print(request,"is the req")
    # Check if the request contains JSON data
    try:
        # Extract the 'article' from the JSON request
        data = request.get_json()
        article = data['article']
        print(article)
        
        # Perform summarization
        summary = summarizer(article, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        
        # Return the summarized text as JSON response
        return jsonify({'summary': summary})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Route to check if the model is running with a sample text
@app.route('/check_model', methods=['GET'])
def check_model():
    sample_text = "This is a sample text to test the summarization model."
    try:
        summary = summarizer(sample_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        return jsonify({'message': 'Model is running successfully.', 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)})

# Route to check if the server is running
@app.route('/check_server', methods=['GET'])
def check_server():
    return "Server is running."

if __name__ == '__main__':
    # Run the Flask app on port 8080
    app.run(debug=True, port=8080)
