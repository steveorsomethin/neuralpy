__kernel void initialize_sums(
        __global float *biases,
        __global float *sums
    ) {
    int gid = get_global_id(0);

    sums[gid] = biases[gid];
}

__kernel void calculate_sums(
        __global float *inputs,
        __global float *weights,
        __global float *sums,
        int input_size
    ) {
    int output_index = get_global_id(0);
    int input_index = get_global_id(1);
    int weight_index = mad24(input_size, output_index, input_index);

    sums[output_index] += inputs[input_index] * weights[weight_index];

    // printf("Sums: %f, %f, %f\n", sums[output_index], inputs[input_index], weights[weight_index]);
}

__kernel void generate_output(
        __global float *sums,
        __global float *outputs
    ) {
    int gid = get_global_id(0);
    float sum = sums[gid];

    outputs[gid] = 1.0f / (1.0f + exp(-sum));

    // printf("Outputs: %f\n", outputs[gid]);
}

__kernel void aggregate_errors(
        __global float *weights,
        __global float *deltas,
        __global float *errors,
        int input_size
    ) {
    int output_index = get_global_id(0);
    int input_index = get_global_id(1);
    int weight_index = mad24(input_size, output_index, input_index);
    
    float delta = deltas[output_index];
    float weight = weights[weight_index];

    errors[input_index] += delta * weight;

    // printf("Errors: %f, %f\n", errors[input_index], delta);
}

__kernel void calculate_deltas(
        __global float *deltas,
        __global float *errors,
        __global float *nodes
    ) {
    int gid = get_global_id(0);
    
    float node = nodes[gid];
    float error = errors[gid];

    deltas[gid] = error * node * (1.0f - node);
    errors[gid] = 0.0f;

    // printf("Deltas: %f %f, %f\n", node, error, deltas[gid]);
}

__kernel void calculate_deltas_output_layer(
        __global float *deltas,
        __global float *errors,
        __global float *outputs,
        __global float *targets
    ) {
    int gid = get_global_id(0);

    float output = outputs[gid];
    float target = targets[gid];
    float error = target - output;

    errors[gid] = error;
    deltas[gid] = error * output * (1.0f - output);

    // printf("Deltas: %f %f, %f\n", output, error, deltas[gid]);
}

__kernel void adjust_weights(
        __global float *inputs,
        __global float *outputs,
        __global float *deltas,
        __global float *changes,
        __global float *weights,
        int input_size,
        float learning_rate,
        float momentum
    ) {
    int output_index = get_global_id(0);
    int input_index = get_global_id(1);
    int grid_index = mad24(input_size, output_index, input_index);

    float input = inputs[input_index];
    float delta = deltas[output_index];
    float change = changes[grid_index];

    change = (learning_rate * delta * input) + (momentum * change);

    changes[grid_index] = change;
    weights[grid_index] += change;

    // printf("Weights: %f, %f, %f\n", input, changes[grid_index], weights[grid_index]);
}

__kernel void adjust_biases(
        __global float *biases,
        __global float *deltas,
        __global float *sums,
        float learning_rate
    ) {
    int gid = get_global_id(0);
    
    biases[gid] += learning_rate * deltas[gid];

    // printf("Biases: %f\n", biases[gid]);
}
