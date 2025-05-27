import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def ejecutar_backprop(
    inputs: np.ndarray = np.array([[0,0],[0,1],[1,0],[1,1]]),
    expected_output: np.ndarray = np.array([[0],[1],[1],[0]]),
    epochs: int = 10000,
    lr: float = 0.1,
    hidden_neurons: int = 2
):
    input_neurons, output_neurons = inputs.shape[1], expected_output.shape[1]
    hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
    hidden_bias    = np.random.uniform(size=(1, hidden_neurons))
    output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
    output_bias    = np.random.uniform(size=(1, output_neurons))

    for _ in range(epochs):
        h_act = np.dot(inputs, hidden_weights) + hidden_bias
        h_out = sigmoid(h_act)
        o_act = np.dot(h_out, output_weights) + output_bias
        o_out = sigmoid(o_act)

        
        o_err = expected_output - o_out
        o_delta = o_err * sigmoid_derivative(o_out)

        h_err = o_delta.dot(output_weights.T)
        h_delta = h_err * sigmoid_derivative(h_out)

        output_weights += h_out.T.dot(o_delta) * lr
        output_bias    += np.sum(o_delta, axis=0, keepdims=True) * lr
        hidden_weights += inputs.T.dot(h_delta) * lr
        hidden_bias    += np.sum(h_delta, axis=0, keepdims=True) * lr

    h_out = sigmoid(np.dot(inputs, hidden_weights) + hidden_bias)
    o_out = sigmoid(np.dot(h_out, output_weights) + output_bias)
    return o_out
