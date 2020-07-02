from flask import Flask, request, jsonify
from flask_cors import CORS
import mian
import numpy as np
app = Flask(__name__)
CORS(app)
dieta_input = np.loadtxt("dataset.txt",delimiter=',',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
dieta_output = np.loadtxt("dataset.txt",delimiter=',',usecols=[25])
aux = np.array(dieta_output)
dieta_output_matrix = []
b = []
for i in range(len(aux)):
    b.append(aux[i])
    dieta_output_matrix.append(b)
    b = []


backPropagation = mian.Backprogation(dieta_input,dieta_output_matrix)




@app.route('/dieta', methods=['POST'])
def registerEndpoint():
    body = None
    body = request.get_json(force=True)
    inputData = body["data"]
    inputD = []
    inputD.append(inputData)
    backPropagation.predecir(inputD)

    return jsonify({"result":backPropagation.Resultado()})

if __name__ == "__main__":
    app.run(debug=False, port=5000)
