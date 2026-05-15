import numpy as np

class NeuralNetwork:
    def __init__(self, layer_list, type="classification"):
        self.layer_list = layer_list
        self.layers = []
        self.type = type
        if layer_list[-1] != 1 and type == "regression":
            raise ValueError("Regression neural network can only have 1 output")
        elif layer_list[-1] <= 1 and type == "classification":
            raise ValueError("Classification neural network must have at least 2 outputs")
        for i in range(len(layer_list) - 1):
            scale = np.sqrt(2.0 / layer_list[i]) * 0.7 
            self.layers.append(np.random.randn(layer_list[i + 1], layer_list[i] + 1) * scale)

    @staticmethod
    def softmax(logits):
        max_output = np.max(logits, axis=0, keepdims=True)
        exps = np.exp(logits - max_output)
        return exps / np.sum(exps, axis=0, keepdims=True)
        
    @staticmethod
    def cross_entropy_derivative(logits, target):
        y_hat = NeuralNetwork.softmax(logits)
        return y_hat - target 

    @staticmethod
    def cross_entropy_error(logits, target):
        probs = np.clip(NeuralNetwork.softmax(logits), 1e-15, 1.0)
        
        loss_per_image = -np.sum(target * np.log(probs), axis=0)
        return np.mean(loss_per_image)

    @staticmethod
    def mean_squared_error(output, target):
        mse_per_image = np.mean((output - target) ** 2, axis=0)
        return np.mean(mse_per_image)

    @staticmethod
    def mean_squared_derivative(output, target):
        return 2 * (output - target)

    def see_layers(self):
        for layer in self.layers:
            print(layer)

    def forward_pass(self, data_batch, activation, activation_derivative, error, error_derivative, target_batch=None):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.error = error
        self.error_derivative = error_derivative
        batch_size = data_batch.shape[1]
        output = {}
        ones = np.ones((1, batch_size))

        output["outputs"] = [data_batch]
        output["outputs"].append(activation(self.layers[0] @ np.vstack([data_batch, ones])))
        
        for i in range(1, len(self.layers) - 1):
            output["outputs"].append(activation(self.layers[i] @ np.vstack([output["outputs"][-1], ones])))
            
        output["outputs"].append(self.layers[-1] @ np.vstack([output["outputs"][-1], ones]))
        if target_batch is not None:
            output["error gradient"] = error_derivative(output["outputs"][-1], target_batch)
            output["error"] = error(output["outputs"][-1], target_batch).item()
        return output
        
    def back_prop(self, fpd, learning_rate):
        batch_size = fpd["outputs"][0].shape[1]
        ones = np.ones((1, batch_size))
        delE_delS = fpd["error gradient"]
        for i in range(len(self.layers) - 1, -1, -1):
            old_weights = self.layers[i].copy()
            delE_delWi = np.clip((delE_delS @ np.vstack([fpd["outputs"][i], ones]).T) / batch_size, -1, 1)
            self.layers[i] = self.layers[i] - (learning_rate * delE_delWi)
            if i != 0:
                delE_delOi = old_weights[:, :-1].T @ delE_delS
                delE_delS = delE_delOi * self.activation_derivative(fpd["outputs"][i])

    def train(self, epochs, dataset, targets, activation, activation_derivative, learning_rate, batch_size=32):
        curr_lr = learning_rate
        for epoch in range(epochs):
            batch_count = 0
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            dataset = dataset[indices]
            targets = targets[indices]
            total_error = 0
            if (epoch + 1) % 50 == 0:
                curr_lr *= 0.5
            for i in range(0, len(dataset), batch_size):
                batch_count += 1
                x_batch = dataset[i : i + batch_size].T
                y_labels = targets[i : i + batch_size]
                if self.type == "regression":
                    y_batch = y_labels.reshape(1, -1)
                else:
                    y_batch = np.eye(self.layer_list[-1])[y_labels].T
                d = self.forward_pass(
                    x_batch, 
                    activation, 
                    activation_derivative,
                    NeuralNetwork.cross_entropy_error 
                        if self.type == "classification" 
                        else NeuralNetwork.mean_squared_error, 
                    NeuralNetwork.cross_entropy_derivative 
                        if self.type == "classification" 
                        else NeuralNetwork.mean_squared_derivative,
                    y_batch
                )
                self.back_prop(d, curr_lr)
                total_error += d["error"]
            if (epoch + 1) % 20 == 0:
                print(f"Avg Error for epoch {epoch + 1}: {total_error / batch_count}")

    def prediction_accuracy(self, dataset, targets):
        if self.type == "classification":
            count = 0
            for i in range(len(dataset)):
                d = self.forward_pass(dataset[i].reshape(-1, 1), 
                                      self.activation, 
                                      self.activation_derivative,
                                      NeuralNetwork.cross_entropy_error, 
                                      NeuralNetwork.cross_entropy_derivative)
                output = NeuralNetwork.softmax(d["outputs"][-1])
                prediction = np.argmax(output, axis=0)
                if prediction == targets[i]:
                    count += 1
            return count / len(dataset)
        else:
            total_squared_error = 0
            total_squared_target = 0
            for i in range(len(dataset)):
                d = self.forward_pass(dataset[i].reshape(-1, 1), 
                      self.activation, 
                      self.activation_derivative,
                      NeuralNetwork.mean_squared_error, 
                      NeuralNetwork.mean_squared_derivative,
                      targets[i])
                total_squared_error += (d["outputs"][-1] - targets[i]) ** 2
                total_squared_target += targets[i] ** 2
            return (total_squared_error / total_squared_target).item()

    def predict(self, dataset):
        if self.type == "classification":
            d = self.forward_pass(dataset.T, 
                  self.activation, 
                  self.activation_derivative,
                  NeuralNetwork.cross_entropy_error, 
                  NeuralNetwork.cross_entropy_derivative)
            probs = NeuralNetwork.softmax(d["outputs"][-1])
            output = np.argmax(probs, axis=0)
        else:
            d = self.forward_pass(dataset.T, 
                  self.activation, 
                  self.activation_derivative,
                  NeuralNetwork.mean_squared_error, 
                  NeuralNetwork.mean_squared_derivative)
            output = d["outputs"][-1]
        return output.T