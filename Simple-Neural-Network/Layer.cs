namespace Simple_Neural_Network;

public class Layer
{
    public Neuron[] Neurons { get; private set; }

    public Layer(int numberOfNeurons, int inputsPerNeuron)
    {
        Neurons = new Neuron[numberOfNeurons];
        for (int i = 0; i < numberOfNeurons; i++)
        {
            Neurons[i] = new Neuron(inputsPerNeuron);
        }
    }

    private Layer(Neuron[] neurons)
    {
        Neurons = new Neuron[neurons.Length];
        for (int i = 0; i < neurons.Length; i++)
        {
            Neurons[i] = neurons[i].Clone();
        }
    }

    public double[] ForwardPass(double[] inputs)
    {
        double[] outputs = new double[Neurons.Length];
        for (int i = 0; i < Neurons.Length; i++)
        {
            outputs[i] = Neurons[i].ForwardPass(inputs);
        }
        return outputs;
    }

    public Layer Clone()
    {
        return new Layer(Neurons);
    }
}



public class Network
{
    public Layer[] Layers { get; private set; }

    public Network(int[] neuronsPerLayer)
    {
        Layers = new Layer[neuronsPerLayer.Length - 1];
        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i] = new Layer(neuronsPerLayer[i + 1], neuronsPerLayer[i]);
        }
    }

    private Network(Layer[] layers)
    {
        Layers = new Layer[layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            Layers[i] = layers[i].Clone();
        }
    }

    public double[] ForwardPass(double[] inputs)
    {
        double[] outputs = inputs;
        foreach (var layer in Layers)
        {
            outputs = layer.ForwardPass(outputs);
        }
        return outputs;
    }

    public void Backpropagate(double[] expectedOutputs, double learningRate)
    {
        // Start by assuming the output layer's output is set and calculating the gradient of the loss
        // with respect to each neuron's output in the output layer.
        Layer outputLayer = Layers[Layers.Length - 1];
        for (int i = 0; i < outputLayer.Neurons.Length; i++)
        {
            Neuron neuron = outputLayer.Neurons[i];
            // Calculate the gradient of the loss with respect to the output (delta)
            neuron.Gradient = (neuron.Output - expectedOutputs[i]) * DerivativeReLU(neuron.Output);
        }

        // Propagate gradients backwards from output to input layer
        for (int layerIndex = Layers.Length - 2; layerIndex >= 0; layerIndex--)
        {
            Layer currentLayer = Layers[layerIndex];
            Layer nextLayer = Layers[layerIndex + 1];

            for (int j = 0; j < currentLayer.Neurons.Length; j++)
            {
                Neuron neuron = currentLayer.Neurons[j];
                double gradientSum = 0;
                for (int k = 0; k < nextLayer.Neurons.Length; k++)
                {
                    gradientSum += nextLayer.Neurons[k].Weights[j] * nextLayer.Neurons[k].Gradient;
                }
                // Update current neuron gradient using derivative of activation function
                neuron.Gradient = gradientSum * DerivativeReLU(neuron.Output);
            }
        }

        // Update weights and biases for all neurons in each layer
        foreach (var layer in Layers)
        {
            foreach (var neuron in layer.Neurons)
            {
                // updates weights and biases based on the gradients calculated
                neuron.UpdateWeights(learningRate);
            }
        }
    }

    private double DerivativeReLU(double x)
    {
        // Derivative of ReLU is 1 for x > 0, otherwise 0
        return x > 0 ? 1 : 0;
    }
    public Network Clone()
    {
        return new Network(Layers);
    }
}

