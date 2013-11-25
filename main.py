import pyopencl as cl
import numpy as np
import numpy.linalg as la
import math

mf = cl.mem_flags

class PairIterator():
    def __init__(self, lst):
        self.lst = lst

    def __iter__(self):
        list_len = len(self.lst)
        for i in range(0, list_len + 1):
            if i == 0:
                yield None, self.lst[i]
            else:
                yield self.lst[i - 1], self.lst[i] if i < list_len else None

    def __reversed__(self):
        result = [(n, p) for p, n in self]
        result.reverse()
        return result

def pair(lst):
    return PairIterator(lst)

class Layer:
    def __init__(self, size, prev_size):
        self.size = size
        self.nodes = np.zeros(size, np.float32)
        self.deltas = np.zeros(size, np.float32)
        self.errors = np.zeros(size, np.float32)

        prev_size = prev_size or 0

        if prev_size > 0:
            self.expected = np.zeros(size, np.float32)
            self.biases = np.random.rand(size).astype(np.float32)
            # for i, _ in enumerate(self.biases):
            #     self.biases[i] = 0.1234
            self.sums = np.zeros(size, np.float32)
            self.weights = np.random.rand(size * prev_size).astype(np.float32)
            # for i, _ in enumerate(self.weights):
            #     self.weights[i] = 0.1234
            self.changes = np.zeros(size * prev_size, np.float32)
        else:
            self.expected = None
            self.biases = None
            self.sums = None
            self.weights = None
            self.changes = None

class InputSizeException(Exception):
    def __init__(self, input_len, layer_len):
        self.input_len = input_len
        self.layer_len = layer_len

    def __str__(self):
        output = "Input size %(input_len)d does not match network input size %(layer_len)d" % \
            {"input_len": self.input_len, "layer_len": self.layer_len}
        return repr(output)

class BasicSolver:
    def prepare_layers(self, layers):
        return None

    def set_target(self, target):
        self.target = target

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_momentum(self, momentum):
        self.momentum = momentum

    def set_input(self, layer, input):
        for i, value in enumerate(input):
            layer.nodes[i] = value

    def feed_forward(self, input_layer, output_layer):
        result = None
        input_len = len(input_layer.nodes)

        for i, _ in enumerate(output_layer.nodes):
            sum = output_layer.biases[i]
            for j, input in enumerate(input_layer.nodes):
                weight_index = (input_len * i) + j
                sum += output_layer.weights[weight_index] * input

            result = 1.0 / (1.0 + math.exp(-sum))

            output_layer.nodes[i] = result

        return result

    def calculate_deltas(self, input_layer, output_layer):
        target = self.target
        output_len = len(output_layer.nodes)

        for i, node in enumerate(output_layer.nodes):
            error = 0.0

            if input_layer == None:
                error = target[i] - node
            else:
                for j, delta in enumerate(input_layer.deltas):
                    weight_index = (output_len * j) + i
                    error += delta * input_layer.weights[weight_index]

            output_layer.errors[i] = error
            output_layer.deltas[i] = error * node * (1 - node)

    def adjust_weights(self, input_layer, output_layer):
        input_len = len(input_layer.nodes)
        learning_rate = self.learning_rate
        momentum = self.momentum

        for i, delta in enumerate(output_layer.deltas):
            for j, node in enumerate(input_layer.nodes):
                change_index = (input_len * i) + j
                change = output_layer.changes[change_index]

                change = (learning_rate * delta * node) + (momentum * change)

                output_layer.changes[change_index] = change
                output_layer.weights[change_index] += change

            output_layer.biases[i] += learning_rate * delta

class CLSolver:
    def __init__(self):
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)
        self.target_buffer = None

        with open("neuralnet.cl", 'r') as fin:
             self.program = cl.Program(self.context, fin.read()).build()

    def prepare_layers(self, layers):
        for layer in layers:
            self.ensure_layer_buffers(layer)

    def buffer(self, hostbuf):
        return cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hostbuf)

    def ensure_layer_buffers(self, layer):
        layer.node_buffer = self.buffer(layer.nodes)
        layer.delta_buffer = self.buffer(layer.deltas)
        layer.error_buffer = self.buffer(layer.errors)

        if layer.biases is not None:
            for i, b in enumerate(layer.biases):
                layer.sums[i] = b

            layer.bias_buffer = self.buffer(layer.biases)
            layer.sum_buffer = self.buffer(layer.sums)
            layer.weight_buffer = self.buffer(layer.weights)
            layer.change_buffer = self.buffer(layer.changes)

    def set_target(self, target):
        flags = mf.READ_ONLY | mf.COPY_HOST_PTR
        self.target = np.array(target, np.float32)

        if self.target_buffer is None:
            self.target_buffer = self.buffer(self.target)
        else:
            cl.enqueue_write_buffer(self.queue, self.target_buffer, self.target)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_momentum(self, momentum):
        self.momentum = momentum

    def set_input(self, layer, input):
        for i, value in enumerate(input):
            layer.nodes[i] = value

        cl.enqueue_write_buffer(self.queue, layer.node_buffer, layer.nodes)

    def feed_forward(self, input_layer, output_layer):
        self.program.initialize_sums(
            self.queue,
            output_layer.nodes.shape,
            None,
            output_layer.bias_buffer,
            output_layer.sum_buffer
        )

        self.program.calculate_sums(
            self.queue,
            (output_layer.size, input_layer.size),
            None,
            input_layer.node_buffer,
            output_layer.weight_buffer,
            output_layer.sum_buffer,
            np.int32(input_layer.size)
        )

        self.program.generate_output(
            self.queue,
            output_layer.nodes.shape,
            None,
            output_layer.sum_buffer,
            output_layer.node_buffer
        )

        cl.enqueue_copy(self.queue, output_layer.nodes, output_layer.node_buffer)

        return output_layer.nodes[len(output_layer.nodes) - 1]

    def calculate_deltas(self, input_layer, output_layer):
        if input_layer == None:
            self.program.calculate_deltas_output_layer(
                self.queue,
                output_layer.nodes.shape,
                None,
                output_layer.delta_buffer,
                output_layer.error_buffer,
                output_layer.node_buffer,
                self.target_buffer
            )
        else:
            self.program.aggregate_errors(
                self.queue,
                (input_layer.size, output_layer.size),
                None,
                input_layer.weight_buffer,
                input_layer.delta_buffer,
                output_layer.error_buffer,
                np.int32(output_layer.size)
            )

            self.program.calculate_deltas(
                self.queue,
                output_layer.nodes.shape,
                None,
                output_layer.delta_buffer,
                output_layer.error_buffer,
                output_layer.node_buffer
            )

        cl.enqueue_copy(self.queue, output_layer.errors, output_layer.error_buffer)

    def adjust_weights(self, input_layer, output_layer):
        self.program.adjust_weights(
            self.queue,
            (output_layer.size, input_layer.size),
            None,
            input_layer.node_buffer,
            output_layer.node_buffer,
            output_layer.delta_buffer,
            output_layer.change_buffer,
            output_layer.weight_buffer,
            np.int32(input_layer.size),
            np.float32(self.learning_rate),
            np.float32(self.momentum)
        )

        self.program.adjust_biases(
            self.queue,
            output_layer.nodes.shape,
            None,
            output_layer.bias_buffer,
            output_layer.delta_buffer,
            output_layer.sum_buffer,
            np.float32(self.learning_rate)
        )

class NeuralNet:
    def __init__(self, sizes, solver = BasicSolver(), learning_rate = 0.3, momentum = 0.1):
        self.solver = solver
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.layers = [Layer(n, p) for p, n in pair(sizes) if n != None]
        self.input_layer = self.layers[0];
        self.output_layer = self.layers[len(self.layers) - 1]

        solver.prepare_layers(self.layers)
        solver.set_learning_rate(learning_rate)
        solver.set_momentum(momentum)

    def run(self, input):
        result = None
        input_layer = self.input_layer
        input_len = len(input)
        layer_len = len(input_layer.nodes)

        if input_len != layer_len:
            raise InputSizeException(input_len, layer_len)

        self.solver.set_input(input_layer, input)

        for p, n in pair(self.layers):
            if p != None and n != None:
                result = self.solver.feed_forward(p, n)

        return result

    def train(self, data):
        iterations = 20000
        error_threshold = 0.00075
        error = 1.0
        i = 0

        while i < iterations and error > error_threshold:
            sum = 0.0
            for input, target in data:
                error = self.train_pattern(input, target)
                sum += error

            error = sum / len(data)

            i += 1

        print "Completed with %d iterations and %f error" % (i, error)

    def train_pattern(self, input, target):
        output_layer = self.output_layer

        self.run(input)
        self.calculate_deltas(target)
        self.adjust_weights()

        return self.mse(self.output_layer.errors)

    def calculate_deltas(self, target):
        self.solver.set_target(target)

        for p, n in reversed(pair(self.layers)):
            if n != None:
                self.solver.calculate_deltas(p, n)

    def adjust_weights(self):
        for p, n in pair(self.layers):
            if p != None and n != None:
                self.solver.adjust_weights(p, n)

    def mse(self, errors):
        sum = 0.0

        for error in errors:
            sum += math.pow(error, 2)
          
        return sum / len(errors);

net1 = NeuralNet([2, 3, 1], BasicSolver(), 3.0, 0.3)
net2 = NeuralNet([2, 3, 1], CLSolver(), 3.0, 0.3)

data = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 1.0], [0.0]),
    ([1.0, 0.0], [1.0])
]

print "Basic"
net1.train(data)
print net1.run([0.0, 0.0])
print net1.run([0.0, 1.0])
print net1.run([1.0, 1.0])
print net1.run([1.0, 0.0])

print "CL"
net2.train(data)
print net2.run([0.0, 0.0])
print net2.run([0.0, 1.0])
print net2.run([1.0, 1.0])
print net2.run([1.0, 0.0])