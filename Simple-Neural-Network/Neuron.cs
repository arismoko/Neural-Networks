using System;

namespace Simple_Neural_Network
{
    public class Neuron
    {
        public double[] Weights { get; private set; }
        public double Bias { get; private set; }
        public double[] Inputs { get; private set; }
        public double Output { get; private set; }
        public double Gradient { get; set; }

        private static Random random = new Random();

        public Neuron(int lengthOfInputs)
        {
            Bias = random.NextDouble() * 0.1;
            Weights = new double[lengthOfInputs];
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = random.NextDouble() * 0.1;
            }
        }

        private Neuron(double[] weights, double bias)
        {
            Weights = (double[])weights.Clone();
            Bias = bias;
        }

        public double ForwardPass(double[] inputs)
        {
            Inputs = inputs;
            double weightedSum = 0;
            for (int i = 0; i < Inputs.Length; i++)
            {
                weightedSum += Weights[i] * Inputs[i];
            }
            weightedSum += Bias;
            Output = ReLU(weightedSum);
            return Output;
        }

        private double ReLU(double x)
        {
            return Math.Max(0, x);
        }

        public void UpdateWeights(double learningRate)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] -= learningRate * Gradient * Inputs[i];
            }
            Bias -= learningRate * Gradient;
        }

        public Neuron Clone()
        {
            return new Neuron(Weights, Bias);
        }
    }

}

