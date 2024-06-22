namespace Simple_Neural_Network;
using Raylib_cs;
using System;
using System.Collections.Generic;
class Program
{
    static void Main(string[] args)
    {
        int numAgents = 5;

        FlappyBirdEnvironment env = new FlappyBirdEnvironment(numAgents);

        int[] networkStructure = new int[] { env.StateSpace, 24, 24, env.ActionSpace };

        List<Agent> agents = new List<Agent>();
        for (int i = 0; i < numAgents; i++)
        {
            agents.Add(new Agent(networkStructure, new ReplayBuffer(10000), .99, learningRate: 0.01));
        }

        int episodes = 1000;
        int updateTargetNetworkSteps = 10;

        for (int e = 0; e < episodes; e++)
        {
            env.Reset();
            List<double> totalRewards = new List<double>(new double[numAgents]);
            bool done = false;
            while (!done)
            {
                env.UpdateEnvironment();
                for (int i = 0; i < numAgents; i++)
                {

                    if (!env.agentDone[i])
                    {
                        int action = agents[i].Act(env.GetState(i));
                        var (nextState, reward, isDone) = env.Step(i, action);
                        totalRewards[i] += reward;

                        agents[i].StoreExperience(env.GetState(i), action, reward, nextState, isDone);
                        agents[i].Train();

                        if (isDone)
                        {
                            Console.WriteLine($"Agent {i + 1}, Episode {e + 1}/{episodes}, Total Reward: {totalRewards[i]:F2}");
                        }
                    }
                }

                env.Render();

                done = env.agentDone.TrueForAll(d => d);
            }

            if (e % updateTargetNetworkSteps == 0)
            {
                foreach (var agent in agents)
                {
                    agent.UpdateTargetNetwork();
                }
            }
        }

        TestAgent(agents[0], env);
    }

    static void TestAgent(Agent agent, FlappyBirdEnvironment env)
    {
        for (int i = 0; i < 10; i++)
        {
            env.Reset();
            double totalReward = 0;
            bool done = false;
            while (!done)
            {
                int action = agent.Act(env.GetState(0));
                var (nextState, reward, isDone) = env.Step(0, action);
                totalReward += reward;

                env.Render();

                if (isDone)
                {
                    Console.WriteLine($"Test Episode {i + 1}/10, Total Reward: {totalReward}");
                    break;
                }
            }
        }
    }
}
public class FlappyBirdEnvironment : IEnvironment
{
    private int screenWidth = 800;
    private int screenHeight = 450;
    private int birdRadius = 20;
    private double gravity = 0.5;
    private int jumpStrength = -10;
    private int pipeWidth = 80;
    private int pipeGap = 200;

    private int pipeX;
    private int pipeHeight;

    public int ActionSpace => 2; // Flap or do nothing
    public int StateSpace => 6; // Example state space: [birdY, birdSpeedY, pipeX, pipeHeight]

    public List<int> birdX;
    public List<int> birdY;
    public List<double> birdSpeedY;
    public List<bool> agentDone;
    public List<double> agentScores;

    private double startTime;
    private double[] currentTimes;

    public FlappyBirdEnvironment(int numAgents)
    {
        Raylib.InitWindow(screenWidth, screenHeight, "Flappy Bird");
        Raylib.SetTargetFPS(60);

        birdX = new List<int>(numAgents);
        birdY = new List<int>(numAgents);
        birdSpeedY = new List<double>(numAgents);
        agentDone = new List<bool>(numAgents);
        agentScores = new List<double>(numAgents);
        startTime = Raylib.GetTime();
        currentTimes = new double[numAgents];
        for (int i = 0; i < numAgents; i++)
        {
            birdX.Add(screenWidth / 4);
            birdY.Add(screenHeight / 2);
            birdSpeedY.Add(0);
            agentDone.Add(false);
            agentScores.Add(0);
        }
    }

    public double[] Reset()
    {
        startTime = Raylib.GetTime();
        currentTimes = new double[birdX.Count];
        for (int i = 0; i < birdX.Count; i++)
        {
            birdX[i] = screenWidth / 4;
            birdY[i] = screenHeight / 2;
            birdSpeedY[i] = 0;
            agentDone[i] = false;
            agentScores[i] = 0;
        }

        pipeX = screenWidth;
        pipeHeight = Raylib.GetRandomValue(100, screenHeight - pipeGap - 100);
        return GetState(0); // Return state for the first agent as an example
    }
    public void UpdateEnvironment()
    {
        pipeX -= 5;
        if (pipeX < -pipeWidth)
        {
            pipeX = screenWidth;
            pipeHeight = Raylib.GetRandomValue(100, screenHeight - pipeGap - 100);
        }
    }
    public (double[] nextState, double reward, bool done) Step(int agentIndex, int action)
    {
        if (action == 1) // Flap
        {
            birdSpeedY[agentIndex] = jumpStrength;
        }

        birdSpeedY[agentIndex] += gravity;
        birdY[agentIndex] += (int)birdSpeedY[agentIndex];



        bool collision = (birdY[agentIndex] - birdRadius < 0) || (birdY[agentIndex] + birdRadius > screenHeight) ||
                         (birdX[agentIndex] + birdRadius > pipeX && birdX[agentIndex] - birdRadius < pipeX + pipeWidth) &&
                         (birdY[agentIndex] - birdRadius < pipeHeight || birdY[agentIndex] + birdRadius > pipeHeight + pipeGap);

        if (collision)
        {
            agentDone[agentIndex] = true;
        }
        else
        {
            currentTimes[agentIndex] = Raylib.GetTime();
        }


        double reward = agentDone[agentIndex] ? -10.0 : 0.1; // Reward for surviving, negative reward for collision
        agentScores[agentIndex] += reward;

        return (GetState(agentIndex), reward, agentDone[agentIndex]);
    }

    public double[] GetState(int agentIndex)
    {
        // Calculate the horizontal distance from the bird to the next pipe
        double distanceToPipe = pipeX - birdX[agentIndex];
        double timeElapsed = currentTimes[agentIndex] - startTime;
        return new double[] {
        birdY[agentIndex],           // Bird's vertical position
        birdSpeedY[agentIndex],      // Bird's vertical speed
        pipeX,                       // X position of the closest pipe
        pipeHeight,                  // Height of the closest pipe
        distanceToPipe,               // Distance to the closest pipe
        timeElapsed                     // Time elapsed since the start of the episode
    };
    }


    public void Render()
    {
        Raylib.BeginDrawing();
        Raylib.ClearBackground(Color.SkyBlue);

        // Draw birds
        for (int i = 0; i < birdX.Count; i++)
        {
            Raylib.DrawCircle(birdX[i], birdY[i], birdRadius, Color.Yellow);
        }

        // Draw pipes
        Raylib.DrawRectangle(pipeX, 0, pipeWidth, pipeHeight, Color.Green);
        Raylib.DrawRectangle(pipeX, pipeHeight + pipeGap, pipeWidth, screenHeight - pipeHeight - pipeGap, Color.Green);

        // Draw score
        for (int i = 0; i < birdX.Count; i++)
        {
            Raylib.DrawText($"Score {i + 1}: {agentScores[i]:F2} Time: {currentTimes[i] - startTime}", 5, 5 + i * 10, 5, Color.Black);
        }

        if (agentDone.TrueForAll(d => d))
        {
            Raylib.DrawText("GAME OVER", screenWidth / 2 - 100, screenHeight / 2 - 50, 50, Color.Black);
            Raylib.DrawText("Press R to Restart", screenWidth / 2 - 120, screenHeight / 2 + 10, 20, Color.Black);
        }

        Raylib.EndDrawing();
    }
}