# Simple Neural Network for Flappy Bird

## Overview

This project implements a simple neural network-based agent to play a Flappy Bird-inspired game environment. The goal is to train multiple agents to navigate through the environment and achieve the highest score possible by avoiding obstacles.

The project is implemented in C# using the [Raylib](https://www.raylib.com/) library for rendering and visualization.

## Features

- **Neural Network Agents**: Each agent is equipped with a simple feedforward neural network that learns to play the game through reinforcement learning.
- **Replay Buffer**: Agents use a replay buffer to store and learn from past experiences.
- **Training and Testing**: The project includes both training and testing phases, allowing you to observe how well the agents have learned.
- **Multiple Agents**: The environment supports multiple agents training simultaneously.
- **Customizable Parameters**: You can adjust the number of agents, network structure, learning rate, and other hyperparameters.

## Project Structure

- **`Program.cs`**: The main entry point for the application. This file contains the logic for training and testing the agents.
- **`FlappyBirdEnvironment.cs`**: Defines the game environment in which the agents operate. It includes the rendering logic, physics simulation, and interaction with agents.
- **`Agent.cs`**: Defines the agent class, including the neural network, decision-making process, and training logic.
- **`ReplayBuffer.cs`**: Implements the replay buffer that stores the agent's experiences for training.

## Dependencies

- **Raylib-cs**: A C# binding for Raylib, used for graphics and window management.
- **.NET Core 3.1 or later**: The project is built on .NET Core, so you'll need a compatible version installed.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Simple_Neural_Network.git
   cd Simple_Neural_Network
   ```

2. **Install dependencies:**

   Make sure you have .NET Core and Raylib installed.

3. **Build and run the project:**

   ```bash
   dotnet run
   ```

## Usage

- **Training**: The agents are trained for a specified number of episodes (`episodes` variable in `Program.cs`). During training, the agents learn to navigate through the game environment by avoiding obstacles.
- **Testing**: After training, you can test the performance of the trained agents in the environment.
- **Rendering**: The environment is rendered using Raylib, allowing you to visually observe the agents' performance.

## Customization

You can customize various aspects of the project:

- **Network Structure**: Modify the `networkStructure` array in `Program.cs` to change the architecture of the neural network.
- **Hyperparameters**: Adjust the learning rate, discount factor, and replay buffer size in `Program.cs`.
- **Number of Agents**: Change the `numAgents` variable to train multiple agents simultaneously.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request with your changes.

Feel free to adjust the details or add any additional information specific to your project!
