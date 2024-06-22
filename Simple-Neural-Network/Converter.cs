namespace Simple_Neural_Network;
public static class RandomProvider
{
    private static Random _random = new Random();
    public static Random GetRandom() => _random;
}

public static class Activators
{
    public static double[] Softmax(double[] logits)
    {
        double maxLogit = double.NegativeInfinity;
        foreach (var logit in logits)
        {
            if (logit > maxLogit)
            {
                maxLogit = logit;
            }
        }

        double sumExp = 0.0;
        foreach (var logit in logits)
        {
            sumExp += Math.Exp(logit - maxLogit);
        }

        double[] probabilities = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            probabilities[i] = Math.Exp(logits[i] - maxLogit) / sumExp;
        }
        return probabilities;
    }

    public static int ArgMax(double[] array)
    {
        int bestIndex = 0;
        double bestValue = array[0];
        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > bestValue)
            {
                bestIndex = i;
                bestValue = array[i];
            }
        }
        return bestIndex;
    }
}
