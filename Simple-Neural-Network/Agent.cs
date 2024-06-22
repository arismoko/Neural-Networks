namespace Simple_Neural_Network;
using System.Collections.Concurrent;
using System.Collections.Generic;
public class Agent
{
    public Network QNetwork;
    public Network TargetNetwork;
    private double gamma;
    private double epsilon;
    private double epsilonDecay;
    private double epsilonMin;
    private double learningRate;
    private int batchSize;
    private ReplayBuffer replayBuffer;
    private bool linearEpsilonDecay;

    public Agent(int[] networkStructure, ReplayBuffer sharedReplayBuffer, double gamma = 0.99, double epsilon = 1.0, double epsilonDecay = 0.995, double epsilonMin = 0.01, double learningRate = 0.001, int batchSize = 32, bool linearEpsilonDecay = false)
    {
        QNetwork = new Network(networkStructure);
        TargetNetwork = QNetwork.Clone();
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.replayBuffer = sharedReplayBuffer;
        this.linearEpsilonDecay = linearEpsilonDecay;
    }

    public int Act(double[] state)
    {
        if (RandomProvider.GetRandom().NextDouble() < epsilon)
        {
            return RandomProvider.GetRandom().Next(QNetwork.Layers[QNetwork.Layers.Length - 1].Neurons.Length);
        }
        var qValues = QNetwork.ForwardPass(state);
        return Array.IndexOf(qValues, qValues.Max());
    }

    public void StoreExperience(double[] state, int action, double reward, double[] nextState, bool done)
    {
        replayBuffer.AddExperience(state, new double[] { action }, reward, nextState, done);
    }

    public void Train()
    {
        if (replayBuffer.Count < batchSize)
            return;

        var minibatch = replayBuffer.Sample(batchSize);

        foreach (var experience in minibatch)
        {
            var state = experience.State;
            var action = (int)experience.Action[0];
            var reward = experience.Reward;
            var nextState = experience.NextState;
            var done = experience.Done;

            double target = reward;
            if (!done)
            {
                target += gamma * TargetNetwork.ForwardPass(nextState).Max();
            }

            var qValues = QNetwork.ForwardPass(state);
            var expectedQValues = (double[])qValues.Clone();
            expectedQValues[action] = target;

            QNetwork.Backpropagate(expectedQValues, learningRate);
        }

        epsilon = Math.Max(epsilonMin, linearEpsilonDecay ? epsilon - epsilonDecay : epsilon * epsilonDecay);
    }

    public void UpdateTargetNetwork()
    {
        TargetNetwork = QNetwork.Clone();
    }
}


public interface IEnvironment
{
    double[] Reset(); // Resets the environment and returns the initial state
    (double[] nextState, double reward, bool done) Step(int agentIndex, int action); // Takes an action for a specific agent and returns the next state, reward, and whether the episode is done
    int ActionSpace { get; } // Number of possible actions
    int StateSpace { get; } // Size of the state representation
    void Render(); // Renders the environment
    double[] GetState(int agentIndex); // Gets the state for a specific agent
}

public class ReplayBuffer
{
    private ConcurrentQueue<Experience> buffer;
    private int maxSize;

    public ReplayBuffer(int maxSize)
    {
        this.maxSize = maxSize;
        buffer = new ConcurrentQueue<Experience>();
    }

    public void AddExperience(double[] state, double[] action, double reward, double[] nextState, bool done)
    {
        if (buffer.Count >= maxSize)
        {
            buffer.TryDequeue(out _);
        }
        buffer.Enqueue(new Experience(state, action, reward, nextState, done));
    }

    public List<Experience> Sample(int batchSize)
    {
        var sample = new List<Experience>();
        var bufferArray = buffer.ToArray();
        var random = RandomProvider.GetRandom();

        for (int i = 0; i < batchSize; i++)
        {
            sample.Add(bufferArray[random.Next(bufferArray.Length)]);
        }

        return sample;
    }

    public int Count => buffer.Count;
}


public class Experience
{
    public double[] State { get; }
    public double[] Action { get; }
    public double Reward { get; }
    public double[] NextState { get; }
    public bool Done { get; }

    public Experience(double[] state, double[] action, double reward, double[] nextState, bool done)
    {
        State = state;
        Action = action;
        Reward = reward;
        NextState = nextState;
        Done = done;
    }
}