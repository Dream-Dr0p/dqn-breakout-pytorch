import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import cv2
import os
import ale_py
from collections import deque
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. 环境创建与预处理
# ------------------------------------------------------------
def make_env():
    env = gym.make("ALE/Breakout-v5", obs_type="rgb")
    return env

def preprocess_frame(frame):
    """将RGB帧转为灰度并缩放至84x84,归一化"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

class FrameStack:
    """堆叠连续k帧,返回形状(k, 84, 84)的张量"""
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, obs):
        processed = preprocess_frame(obs)
        for _ in range(self.k):
            self.frames.append(processed)
        return self.get_state()

    def step(self, obs):
        processed = preprocess_frame(obs)
        self.frames.append(processed)
        return self.get_state()

    def get_state(self):
        return np.stack(self.frames, axis=0)

# ------------------------------------------------------------
# 2. DQN网络结构
# ------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)

# ------------------------------------------------------------
# 3. 经验回放缓冲区
# ------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# ------------------------------------------------------------
# 4. 智能体定义
# ------------------------------------------------------------
class DQNAgent:
    def __init__(self, num_actions, device):
        self.num_actions = num_actions
        self.device = device
        self.policy_net = DQN(4, num_actions).to(device)
        self.target_net = DQN(4, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 500000
        self.steps = 0
        self.target_update_freq = 1000

    def select_action(self, state, evaluation=False):
        if not evaluation and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * \
                       np.exp(-1. * self.steps / self.epsilon_decay)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ------------------------------------------------------------
# 5. 训练主循环（支持断点续训、防卡死）
# ------------------------------------------------------------
def train(num_episodes=500, resume=False, checkpoint_path="checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    frame_stack = FrameStack(k=4)

    agent = DQNAgent(env.action_space.n, device)
    start_episode = 0

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.steps = checkpoint['steps']
        agent.epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode']
        print(f"从第 {start_episode} 个 episode 恢复训练，已训练步数: {agent.steps}")

    total_steps = agent.steps
    episode_rewards = []
    losses = []
    update_every = 4
    max_steps_per_episode = 18000

    for episode in range(start_episode, num_episodes):
        if episode % 100 == 0 and episode != start_episode:
            torch.save({
                'policy_net': agent.policy_net.state_dict(),
                'target_net': agent.target_net.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'steps': agent.steps,
                'epsilon': agent.epsilon,
                'episode': episode,
            }, checkpoint_path)
            print(f"检查点已保存 (episode {episode})")

        obs, info = env.reset()
        state = frame_stack.reset(obs)
        episode_reward = 0
        done = False
        episode_start_step = total_steps

        while not done:
            total_steps += 1
            agent.steps = total_steps
            agent.update_epsilon()
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if total_steps - episode_start_step > max_steps_per_episode:
                done = True

            # 修复1：始终用真实的 next_obs 构造 next_state，终止状态不再被替换为零
            next_state = frame_stack.step(next_obs)

            # 修复2：奖励裁剪（将不同分数统一为 +1, -1, 0，提升训练稳定性）
            clipped_reward = np.sign(reward)

            agent.memory.push(state, action, clipped_reward, next_state, float(done))
            state = next_state
            episode_reward += reward   # 记录原始奖励以评估真实表现

            if total_steps % update_every == 0:
                loss = agent.train_step()
                if loss:
                    losses.append(loss)
            if total_steps % agent.target_update_freq == 0:
                agent.update_target_network()

        episode_rewards.append(episode_reward)
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            loss_info = f", Loss: {np.mean(losses[-200:]):.4f}" if losses else ""
            print(f"Episode {episode}, Steps {total_steps}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}{loss_info}")

    env.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance of DQN on Breakout")
    plt.savefig("breakout_dqn_rewards.png")
    plt.show()

    torch.save(agent.policy_net.state_dict(), "dqn_breakout_model.pth")
    print("模型已保存为 dqn_breakout_model.pth")

# ------------------------------------------------------------
# 6. 评估函数（逻辑简化，移除无意义的黑屏替换）
# ------------------------------------------------------------
def evaluate(model_path="dqn_breakout_model.pth", num_episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    agent = DQNAgent(env.action_space.n, device)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.policy_net.eval()

    stack = FrameStack(4)
    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = stack.reset(obs)
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluation=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if not done:
                state = stack.step(obs)
            # 修复3：done之后不再做黑屏赋值，循环自然退出
            total_reward += reward
        print(f"Episode {ep+1}: 总得分 {total_reward}")
    env.close()

# ------------------------------------------------------------
# 主程序入口
# ------------------------------------------------------------
if __name__ == "__main__":
    # 从头训练 500 个 episode；若要续训，请改成 train(500, resume=True)
    #train(10001, resume=True)

    # 训练完成后可以运行评估（取消下面一行的注释即可）
    evaluate()