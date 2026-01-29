# 🚀 Lunar Lander DQN Agent

A Deep Q-Network (DQN) reinforcement learning agent that learns to safely land a lunar module on the moon using OpenAI's Gymnasium LunarLander-v3 environment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Usage](#usage)
- [Network Architecture](#network-architecture)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a **Deep Q-Network (DQN)** agent that learns to control a lunar lander spacecraft through reinforcement learning. The agent must learn to:

- Navigate to the landing pad
- Control orientation using left/right thrusters
- Manage main engine for controlled descent
- Execute a safe landing while minimizing fuel consumption

The implementation demonstrates advanced RL techniques including experience replay, target networks, and epsilon-greedy exploration.

## ✨ Features

- **Modern DQN Implementation**: Uses PyTorch for neural network approximation of Q-values
- **Experience Replay**: Stores 100,000 transitions for stable learning
- **Target Network**: Separate target network updated every 10 episodes for training stability
- **Epsilon-Greedy Exploration**: Decaying exploration strategy (1.0 → 0.01)
- **GPU Support**: Automatically utilizes CUDA when available
- **Comprehensive Visualization**:
  - Training progress plots with moving averages
  - Animated agent performance demonstrations
- **Educational Documentation**: Detailed theory explanations in the notebook

## 🧠 Algorithm

The agent uses **Deep Q-Learning** with the following components:

### Core Techniques

1. **Q-Learning with Function Approximation**
   - Neural network approximates Q(s, a) instead of maintaining a Q-table
   - Handles continuous 8-dimensional state space efficiently

2. **Experience Replay**
   - Circular buffer stores (state, action, reward, next_state, done) tuples
   - Random mini-batch sampling breaks temporal correlations
   - Improves sample efficiency and learning stability

3. **Target Network**
   - Policy Network: Updated every step during training
   - Target Network: Periodically synchronized (every 10 episodes)
   - Prevents moving target problem in Q-learning

4. **Bellman Equation Update**
   ```
   Q_target = reward + γ * max(Q(s', a')) * (1 - done)
   Loss = MSE(Q_predicted, Q_target)
   ```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install SWIG (required for Box2D)

```bash
# On Ubuntu/Debian
sudo apt-get install swig

# On macOS
brew install swig

# On Windows
# Download from http://www.swig.org/download.html
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install gymnasium[box2d]
pip install torch torchvision
pip install numpy matplotlib
pip install jupyter
```

## 🚀 Usage

### Training the Agent

1. Open the Jupyter notebook:
```bash
jupyter notebook Lunar_lander_agent.ipynb
```

2. Run all cells sequentially to:
   - Set up the environment
   - Initialize the DQN agent
   - Train for 500 episodes
   - Visualize training progress
   - Evaluate the trained agent

### Quick Start

The notebook is organized into clear sections:

1. **Environment Setup**: Initialize LunarLander-v3
2. **Theory**: Q-Learning and DQN fundamentals
3. **Network Definition**: Neural network architecture
4. **Training**: Main training loop with experience replay
5. **Evaluation**: Test the trained agent and generate animations

## 🏗️ Network Architecture

```
Input Layer:     8 neurons  (state: x, y, vx, vy, angle, angular_vel, leg1, leg2)
                     ↓
Hidden Layer 1:  128 neurons + ReLU
                     ↓
Hidden Layer 2:  128 neurons + ReLU
                     ↓
Output Layer:    4 neurons   (Q-values: noop, left, main, right)
```

**Optimizer**: Adam (learning rate: 0.001)
**Loss Function**: Mean Squared Error (MSE)

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GAMMA` | 0.99 | Discount factor for future rewards |
| `LR` | 1e-3 | Learning rate for Adam optimizer |
| `EPSILON_START` | 1.0 | Initial exploration probability |
| `EPSILON_END` | 0.01 | Minimum exploration probability |
| `EPSILON_DECAY` | 500 | Rate of epsilon decay |
| `BATCH_SIZE` | 64 | Mini-batch size for training |
| `MEMORY_SIZE` | 100,000 | Experience replay buffer capacity |
| `EPISODES` | 500 | Number of training episodes |
| `TARGET_UPDATE` | 10 | Episodes between target network updates |

## 📊 Results

The agent successfully learns to land the lunar module after training:

- **Training**: 500 episodes with epsilon-greedy exploration
- **Performance**: Converges to safe landing strategy
- **Visualization**: Animated demonstrations show the agent's learned policy

The notebook includes:
- Training curve showing reward progression
- Moving average (10 episodes) to track learning trends
- Final agent performance animations

## 📁 Project Structure

```
Lunar-lander/
├── Lunar_lander_agent.ipynb  # Main notebook with implementation
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore patterns
```

## 🎓 Learning Resources

This project is ideal for understanding:

- Deep Reinforcement Learning fundamentals
- PyTorch neural network implementation
- Gymnasium (OpenAI Gym) environments
- Experience replay and target networks
- Balancing exploration vs exploitation

The notebook contains detailed explanations of each concept, making it suitable for both learning and reference.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs or issues
- Suggest new features or improvements
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) for the LunarLander environment
- [PyTorch](https://pytorch.org/) for the deep learning framework
- Original DQN paper: [Mnih et al., 2015](https://www.nature.com/articles/nature14236)

---

**Note**: This is an educational project demonstrating DQN implementation. The code is extensively commented and designed for learning purposes.
