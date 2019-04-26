def predict(row, weights):
    threshold = weights[0]
    for i in range(1, len(row)):
        threshold += weights[i] * row[i]
    if threshold >= 0.0:
        return 1.0
    else:
        return 0.0


def train_perceptron(data, learning_rate, epochs):
    weights = []
    for i in range(len(data[0])):
        weights.append(0.0)
    for epoch in range(epochs):
        error_sum = 0.0
        for data_point in data:
            pred = predict(data_point, weights)
            error = data_point[0] - pred
            error_sum += error ** 2
            weights[0] = weights[0] + (learning_rate * error)
            for i in range(1, len(data_point)):
                weights[i] = weights[i] + (learning_rate * error * data_point[i])
    return weights
