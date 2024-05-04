from flask import Flask, request, jsonify
from flask_cors import CORS
from term_deposit import train_model, delete_model, prediction

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return 'Server is up and running!', 200

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    data = request.json
    print(data)
    decision = prediction(data)
    print(decision)
    return jsonify({'decision': decision}), 200

@app.route('/train_model', methods=['GET'])
def retrain_model():
    try:
        train_model()
        return jsonify({'message':'model trained successfully'})
    except Exception as e:
        err = str(e)
        return jsonify({'error-msg' : err})

@app.route('/delete_model', methods=['GET'])
def delete_trained_model():
    try:
        delete_model()
        return jsonify({'message':'model deleted successfully'})
    except Exception as e:
        err = str(e)
        return jsonify({'error-msg' : err})

if __name__ == '__main__':
    app.run(debug=True)
