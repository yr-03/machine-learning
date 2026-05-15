import numpy as np

class NeuralNetwork:
    def __init__(self, layer_list):
        self.layers = []
        for i in range(len(layer_list) - 1):
            self.layers.append(np.random.randn(layer_list[i + 1], layer_list[i] + 1) * 0.01)
    
    def see_layers(self):
        for layer in self.layers:
            print(layer)

    def forward_pass(self, data_point, target, activation, error, error_derivative):
        output = {}
        output["data"] = data_point
        output["layer 1 sum"] = self.layers[0] @ np.vstack([data_point, [1]])
        output["layer 1 output"] = activation(output["layer 1 sum"])
        for i in range(1, len(self.layers)):
            output[f"layer {i+1} sum"] = self.layers[i] @ np.vstack([output[f"layer {i} output"], [1]])
            output[f"layer {i+1} output"] = activation(output[f"layer {i+1} sum"])
        output["error"] = error(output[f"layer {len(self.layers)} output"], target).item()
        output["error gradient"] = error_derivative(output[f"layer {len(self.layers)} output"], target).item()
        return output
        
    def back_prop(self, fpd, learning_rate, activation_derivative):
        delE_delOlast = fpd["error gradient"]
        delE_delSlast = delE_delOlast * activation_derivative(fpd[f"layer {len(self.layers)} output"])
        delE_delWlast = delE_delSlast * np.vstack([fpd[f"layer {len(self.layers) - 1} output"], [1]]).T
        self.layers[-1] = self.layers[-1] - (learning_rate * delE_delWlast)

        for i in range(len(self.layers) - 1, 1, -1):
            delE_delOi = self.layers[i][:, :-1].T @ delE_delSlast
            delE_delSlast = delE_delOi * activation_derivative(fpd[f"layer {i} output"])
            delE_delWi = delE_delSlast @ np.vstack([fpd[f"layer {i-1} output"], [1]]).T
            self.layers[i-1] = self.layers[i-1] - (learning_rate * delE_delWi)

        delE_delO1 = self.layers[1][:, :-1].T @ delE_delSlast
        delE_delS1 = delE_delO1 * activation_derivative(fpd[f"layer 1 output"])
        delE_delW1 = delE_delS1 @ np.vstack([fpd["data"], [1]]).T
        self.layers[0] = self.layers[0] - (learning_rate * delE_delW1)

    def train(self, epochs, dataset, targets, activation, error, error_derivative, learning_rate, activation_derivative):
        for epoch in range(epochs):
            counter = 1
            for i in range(len(dataset)):
                d = self.forward_pass(dataset[i].reshape(-1, 1), targets[i], activation, error, error_derivative)
                self.back_prop(d, learning_rate, activation_derivative)
                counter += 1
                if counter % 50 == 0 and epoch % 50 == 0:
                    print(f"Error: {d["error"]:.6f}")
