namespace Simple_Neural_Network;
using System.Collections.Generic;
public static class Encoder
{
    public static int[] OneHotEncode<T>(T category, T[] allCategories)
    {
        int[] oneHot = new int[allCategories.Length];
        int index = Array.IndexOf(allCategories, category);
        if (index >= 0)
        {
            oneHot[index] = 1;
        }
        return oneHot;
    }
}
