#!/usr/bin/env python3
"""
HeuristiCore: Reinforcement Learning Agent for Binary Pattern Transformation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent learns to compose sequences of logical operations (rules) to transform
binary patterns from current state to target state.

Architecture:
  • Environment: HilbertNet-based execution environment
  • Actor-Critic: Neural networks for rule selection and value estimation
  • UBT Integration: Parameter optimization for selected rules
  • Anti-stagnation: Exploration bonuses and entropy regularization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import time
import random
import pickle
import json
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITIES
# ============================================================================

def popcount(x: int) -> int:
    """Fast bit count."""
    try:
        return x.bit_count()
    except AttributeError:
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return popcount(a ^ b)


# ============================================================================
# SIMPLIFIED HILBERT NET (Embedded)
# ============================================================================

class RuleRegistry:
    """Registry of binary operations."""
    
    def __init__(self):
        self.rules: Dict[str, Callable] = {}
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in rules."""
        self.rules['xor'] = lambda v, p: v ^ p
        self.rules['and'] = lambda v, p: v & p
        self.rules['or'] = lambda v, p: v | p
        self.rules['not'] = lambda v, p: ~v & 0xFFFFFFFF
        self.rules['add'] = lambda v, p: (v + p) & 0xFFFFFFFF
        self.rules['sub'] = lambda v, p: (v - p) & 0xFFFFFFFF
        self.rules['rotate_l'] = lambda v, p: ((v << (p % 32)) | (v >> (32 - (p % 32)))) & 0xFFFFFFFF
        self.rules['rotate_r'] = lambda v, p: ((v >> (p % 32)) | (v << (32 - (p % 32)))) & 0xFFFFFFFF
        self.rules['shift_l'] = lambda v, p: (v << (p % 32)) & 0xFFFFFFFF
        self.rules['shift_r'] = lambda v, p: v >> (p % 32)
    
    def get(self, name: str) -> Callable:
        """Get rule function."""
        if name not in self.rules:
            raise ValueError(f"Unknown rule: {name}")
        return self.rules[name]
    
    def list_rules(self) -> List[str]:
        """List all rule names."""
        return list(self.rules.keys())


class SimplifiedHilbertNet:
    """Simplified HilbertNet for RL environment."""
    
    def __init__(self):
        self.rules = RuleRegistry()
        self.state: int = 0
    
    def apply_rule(self, rule_name: str, parameter: int) -> int:
        """Apply rule to current state."""
        rule_func = self.rules.get(rule_name)
        self.state = rule_func(self.state, parameter) & 0xFFFFFFFF
        return self.state
    
    def set_state(self, state: int):
        """Set current state."""
        self.state = state & 0xFFFFFFFF
    
    def get_state(self) -> int:
        """Get current state."""
        return self.state


# ============================================================================
# UNIVERSAL BINARY TENSOR (Simplified for RL)
# ============================================================================

class RLTransformObjective:
    """Custom UBT objective for RL parameter optimization."""
    
    def __init__(self, current_state: int, target_state: int, rule_func: Callable):
        self.current = current_state
        self.target = target_state
        self.rule_func = rule_func
    
    def evaluate(self, parameter: int) -> float:
        """Evaluate parameter quality (negative distance to target)."""
        try:
            transformed = self.rule_func(self.current, parameter) & 0xFFFFFFFF
            distance = hamming_distance(transformed, self.target)
            return -distance  # UBT maximizes, so negate distance
        except Exception:
            return -32  # Worst possible score


class SimplifiedUBT:
    """Simplified UBT for parameter optimization."""
    
    def __init__(self, bits: int = 32):
        self.bits = bits
    
    def optimize(self, objective: RLTransformObjective, budget: int = 5000) -> int:
        """Optimize parameter using simple genetic algorithm."""
        pop_size = 50
        elite_size = 5
        mutation_rate = 0.05
        
        # Initialize population
        population = [random.getrandbits(self.bits) for _ in range(pop_size)]
        best_param = population[0]
        best_score = objective.evaluate(best_param)
        
        generations = budget // pop_size
        
        for gen in range(generations):
            # Evaluate
            scores = [objective.evaluate(p) for p in population]
            
            # Track best
            gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
            if scores[gen_best_idx] > best_score:
                best_param = population[gen_best_idx]
                best_score = scores[gen_best_idx]
            
            # Early stopping if perfect
            if best_score == 0:
                break
            
            # Selection: keep elite
            elite_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elite_size]
            new_pop = [population[i] for i in elite_indices]
            
            # Reproduction
            while len(new_pop) < pop_size:
                # Tournament selection
                p1 = population[random.choice(elite_indices)]
                p2 = population[random.choice(elite_indices)]
                
                # Crossover
                mask = random.getrandbits(self.bits)
                child = (p1 & mask) | (p2 & ~mask)
                
                # Mutation
                if random.random() < mutation_rate:
                    bit = random.randint(0, self.bits - 1)
                    child ^= (1 << bit)
                
                new_pop.append(child)
            
            population = new_pop[:pop_size]
        
        return best_param


# ============================================================================
# ENVIRONMENT MANAGER
# ============================================================================

@dataclass
class EnvState:
    """Environment state representation."""
    current: int
    target: int
    step: int = 0
    max_steps: int = 10


class EnvironmentManager:
    """RL Environment for binary pattern transformation."""
    
    def __init__(self, max_steps: int = 10):
        self.hilbert_net = SimplifiedHilbertNet()
        self.max_steps = max_steps
        self.state: Optional[EnvState] = None
        self.action_space = self.hilbert_net.rules.list_rules()
    
    def reset(self, current: int, target: int) -> EnvState:
        """Reset environment with new task."""
        self.state = EnvState(current=current, target=target, max_steps=self.max_steps)
        self.hilbert_net.set_state(current)
        return self.state
    
    def step(self, action: Tuple[str, int]) -> Tuple[EnvState, float, bool, Dict]:
        """Execute action and return (new_state, reward, done, info)."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        rule_name, parameter = action
        
        # Store old distance
        old_distance = hamming_distance(self.state.current, self.state.target)
        
        # Apply rule
        new_current = self.hilbert_net.apply_rule(rule_name, parameter)
        
        # Calculate new distance
        new_distance = hamming_distance(new_current, self.state.target)
        
        # Calculate reward
        reward = self._calculate_reward(old_distance, new_distance)
        
        # Update state
        self.state.current = new_current
        self.state.step += 1
        
        # Check if done
        done = (new_distance == 0) or (self.state.step >= self.max_steps)
        
        info = {
            'old_distance': old_distance,
            'new_distance': new_distance,
            'rule': rule_name,
            'improvement': old_distance - new_distance
        }
        
        return self.state, reward, done, info
    
    def _calculate_reward(self, old_distance: int, new_distance: int) -> float:
        """Calculate reward based on distance change."""
        # Main reward: improvement in distance
        improvement = old_distance - new_distance
        reward = improvement * 2.0  # Scale up improvements
        
        # Small penalty for each step (encourage efficiency)
        reward -= 0.1
        
        # Large bonus for perfect match
        if new_distance == 0:
            reward += 50.0
        
        # Penalty for making things worse
        if improvement < 0:
            reward -= 5.0
        
        return reward


# ============================================================================
# NEURAL NETWORKS (Actor-Critic)
# ============================================================================

class SimpleNN:
    """Simple feedforward neural network."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        
        # For gradient computation
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Hidden layer
        z1 = x @ self.w1 + self.b1
        a1 = np.tanh(z1)
        
        # Output layer
        z2 = a1 @ self.w2 + self.b2
        
        # Cache for backprop
        self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2}
        
        return z2
    
    def backward(self, grad_output: np.ndarray, lr: float = 0.001) -> None:
        """Backward pass and update weights."""
        x = self.cache['x']
        a1 = self.cache['a1']
        
        # Clip gradients to prevent overflow
        grad_output = np.clip(grad_output, -10, 10)
        
        # Output layer gradients
        dw2 = a1.T @ grad_output
        db2 = np.sum(grad_output, axis=0)
        
        # Hidden layer gradients
        da1 = grad_output @ self.w2.T
        dz1 = da1 * (1 - a1**2)  # tanh derivative
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        # Clip all gradients
        dw2 = np.clip(dw2, -1, 1)
        db2 = np.clip(db2, -1, 1)
        dw1 = np.clip(dw1, -1, 1)
        db1 = np.clip(db1, -1, 1)
        
        # Update weights
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1


class RuleSelector:
    """Actor network: selects rule based on state."""
    
    def __init__(self, num_rules: int, hidden_size: int = 128):
        # Input: 64 bits (32 for current + 32 for target)
        self.network = SimpleNN(64, hidden_size, num_rules)
        self.num_rules = num_rules
    
    def _state_to_input(self, current: int, target: int) -> np.ndarray:
        """Convert state to network input."""
        current_bits = [(current >> i) & 1 for i in range(32)]
        target_bits = [(target >> i) & 1 for i in range(32)]
        return np.array([current_bits + target_bits], dtype=np.float32)
    
    def get_rule_probs(self, current: int, target: int) -> np.ndarray:
        """Get probability distribution over rules."""
        x = self._state_to_input(current, target)
        logits = self.network.forward(x)[0]
        
        # Clip logits to prevent overflow
        logits = np.clip(logits, -10, 10)
        
        # Softmax with numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-8)
        
        # Ensure valid probabilities
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / np.sum(probs)  # Renormalize
        
        return probs
    
    def sample_rule(self, current: int, target: int) -> int:
        """Sample rule index from distribution."""
        probs = self.get_rule_probs(current, target)
        return np.random.choice(len(probs), p=probs)
    
    def update(self, current: int, target: int, rule_idx: int, advantage: float, lr: float = 0.001):
        """Update network using policy gradient."""
        x = self._state_to_input(current, target)
        
        # Forward pass
        logits = self.network.forward(x)[0]
        logits = np.clip(logits, -10, 10)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-8)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / np.sum(probs)
        
        # Clip advantage to prevent extreme updates
        advantage = np.clip(advantage, -10, 10)
        
        # Compute gradient
        grad = -probs.copy()
        grad[rule_idx] += 1  # Gradient of log(prob)
        grad = grad * advantage  # Scale by advantage
        
        # Reshape for backward pass
        grad = grad.reshape(1, -1)
        
        # Backward pass
        self.network.backward(grad, lr=lr)


class ValueEstimator:
    """Critic network: estimates state value."""
    
    def __init__(self, hidden_size: int = 128):
        self.network = SimpleNN(64, hidden_size, 1)
    
    def _state_to_input(self, current: int, target: int) -> np.ndarray:
        """Convert state to network input."""
        current_bits = [(current >> i) & 1 for i in range(32)]
        target_bits = [(target >> i) & 1 for i in range(32)]
        return np.array([current_bits + target_bits], dtype=np.float32)
    
    def estimate_value(self, current: int, target: int) -> float:
        """Estimate value of state."""
        x = self._state_to_input(current, target)
        value = self.network.forward(x)[0, 0]
        return float(value)
    
    def update(self, current: int, target: int, td_error: float, lr: float = 0.001):
        """Update network using TD error."""
        x = self._state_to_input(current, target)
        
        # Forward pass
        self.network.forward(x)
        
        # Clip TD error to prevent overflow
        td_error = np.clip(td_error, -10, 10)
        
        # Gradient is just the TD error
        grad = np.array([[td_error]], dtype=np.float32)
        
        # Backward pass
        self.network.backward(grad, lr=lr)


class PolicyAgent:
    """Combined Actor-Critic agent with UBT parameter optimization."""
    
    def __init__(self, action_space: List[str]):
        self.action_space = action_space
        self.rule_selector = RuleSelector(len(action_space))
        self.value_estimator = ValueEstimator()
        self.parameter_optimizer = SimplifiedUBT()
        self.hilbert_net = SimplifiedHilbertNet()  # For rule functions
        
        # Entropy coefficient for exploration
        self.entropy_coef = 0.01
    
    def select_action(self, current: int, target: int) -> Tuple[str, int]:
        """Select action: (rule_name, parameter)."""
        # Select rule
        rule_idx = self.rule_selector.sample_rule(current, target)
        rule_name = self.action_space[rule_idx]
        
        # Optimize parameter using UBT
        rule_func = self.hilbert_net.rules.get(rule_name)
        objective = RLTransformObjective(current, target, rule_func)
        parameter = self.parameter_optimizer.optimize(objective, budget=2000)
        
        return rule_name, parameter
    
    def get_value(self, current: int, target: int) -> float:
        """Get estimated value of state."""
        return self.value_estimator.estimate_value(current, target)
    
    def update(self, trajectory: List[Dict], gamma: float = 0.99, lr_actor: float = 0.001, 
               lr_critic: float = 0.001):
        """Update both actor and critic from trajectory."""
        # Compute returns and advantages
        returns = []
        advantages = []
        G = 0
        
        for t in reversed(range(len(trajectory))):
            step = trajectory[t]
            reward = step['reward']
            value = step['value']
            
            G = reward + gamma * G
            returns.insert(0, G)
            
            # Advantage = return - value
            advantage = G - value
            advantages.insert(0, advantage)
        
        # Update networks
        for t, step in enumerate(trajectory):
            current = step['current']
            target = step['target']
            rule_idx = self.action_space.index(step['rule'])
            advantage = advantages[t]
            td_error = returns[t] - step['value']
            
            # Update actor
            self.rule_selector.update(current, target, rule_idx, advantage, lr=lr_actor)
            
            # Update critic
            self.value_estimator.update(current, target, td_error, lr=lr_critic)


# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    """Manages training and evaluation data."""
    
    def __init__(self, mode: str = 'synthetic'):
        self.mode = mode
    
    def generate_synthetic_pair(self) -> Tuple[int, int]:
        """Generate random (current, target) pair."""
        current = random.getrandbits(32)
        target = random.getrandbits(32)
        return current, target
    
    def generate_xor_task(self) -> Tuple[int, int, int]:
        """Generate XOR task: current XOR key = target."""
        current = random.getrandbits(32)
        key = random.getrandbits(32)
        target = current ^ key
        return current, target, key


# ============================================================================
# TRAINING MANAGER
# ============================================================================

class TrainingManager:
    """Manages training loop with anti-stagnation."""
    
    def __init__(self, agent: PolicyAgent, env: EnvironmentManager):
        self.agent = agent
        self.env = env
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.best_reward = -float('inf')
        self.stagnation_counter = 0
    
    def run_episode(self, current: int, target: int) -> Tuple[float, int, List[Dict]]:
        """Run single episode."""
        state = self.env.reset(current, target)
        trajectory = []
        total_reward = 0.0
        
        while True:
            # Get action
            rule_name, parameter = self.agent.select_action(state.current, state.target)
            
            # Get value estimate
            value = self.agent.get_value(state.current, state.target)
            
            # Execute action
            new_state, reward, done, info = self.env.step((rule_name, parameter))
            
            # Record trajectory
            trajectory.append({
                'current': state.current,
                'target': state.target,
                'rule': rule_name,
                'parameter': parameter,
                'reward': reward,
                'value': value,
                'info': info
            })
            
            total_reward += reward
            state = new_state
            
            if done:
                break
        
        return total_reward, len(trajectory), trajectory
    
    def train(self, num_episodes: int = 10000, data_manager: Optional[DataManager] = None):
        """Main training loop."""
        if data_manager is None:
            data_manager = DataManager('synthetic')
        
        logger.info(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Generate task
            current, target = data_manager.generate_synthetic_pair()
            
            # Run episode
            total_reward, length, trajectory = self.run_episode(current, target)
            
            # Update agent
            self.agent.update(trajectory)
            
            # Record metrics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(length)
            
            # Check for improvement
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # Anti-stagnation: increase exploration
            if self.stagnation_counter > 100:
                self.agent.entropy_coef = min(0.1, self.agent.entropy_coef * 1.1)
                self.stagnation_counter = 0
                logger.info(f"Increasing exploration: entropy_coef = {self.agent.entropy_coef:.4f}")
            
            # Logging
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                logger.info(f"Episode {episode+1}/{num_episodes}: "
                           f"avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}, "
                           f"best={self.best_reward:.2f}")
    
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            'total_episodes': len(self.episode_rewards),
            'best_reward': self.best_reward,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0.0,
            'avg_length_last_100': np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else 0.0,
            'entropy_coef': self.agent.entropy_coef
        }


# ============================================================================
# ANALYSIS DASHBOARD
# ============================================================================

class AnalysisDashboard:
    """Analysis and visualization of training."""
    
    def __init__(self, training_manager: TrainingManager):
        self.tm = training_manager
    
    def print_summary(self):
        """Print training summary."""
        stats = self.tm.get_statistics()
        
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Total Episodes:        {stats['total_episodes']}")
        print(f"Best Reward:           {stats['best_reward']:.2f}")
        print(f"Avg Reward (last 100): {stats['avg_reward_last_100']:.2f}")
        print(f"Avg Length (last 100): {stats['avg_length_last_100']:.1f}")
        print(f"Current Entropy Coef:  {stats['entropy_coef']:.4f}")
        print("="*70)
    
    def test_agent(self, num_tests: int = 10):
        """Test agent on XOR tasks."""
        print("\n" + "="*70)
        print("TESTING ON XOR TASKS")
        print("="*70)
        
        data_manager = DataManager('synthetic')
        successes = 0
        
        for i in range(num_tests):
            current, target, true_key = data_manager.generate_xor_task()
            
            # Run episode
            total_reward, length, trajectory = self.tm.run_episode(current, target)
            
            # Check if solved
            final_state = trajectory[-1]['info']['new_distance']
            solved = (final_state == 0)
            
            if solved:
                successes += 1
            
            print(f"Test {i+1}: current={current:08x}, target={target:08x}, "
                  f"solved={solved}, reward={total_reward:.1f}, steps={length}")
            
            # Show discovered sequence
            print(f"  Sequence: ", end="")
            for step in trajectory:
                print(f"{step['rule']}(0x{step['parameter']:08x}) -> ", end="")
            print(f"dist={final_state}")
        
        print(f"\nSuccess Rate: {successes}/{num_tests} ({100*successes/num_tests:.1f}%)")
        print("="*70)


# ============================================================================
# MAIN DEMO
# ============================================================================

def demo_full_system():
    """Complete system demonstration."""
    print("\n" + "#"*70)
    print("#  HeuristiCore: RL Agent for Binary Pattern Transformation")
    print("#  Training agent to learn rule composition...")
    print("#"*70)
    
    # Initialize components
    env = EnvironmentManager(max_steps=10)
    agent = PolicyAgent(env.action_space)
    trainer = TrainingManager(agent, env)
    dashboard = AnalysisDashboard(trainer)
    
    # Train
    print("\nPhase 1: Training on synthetic tasks...")
    trainer.train(num_episodes=1000)
    
    # Analyze
    dashboard.print_summary()
    
    # Test
    print("\nPhase 2: Testing on XOR tasks...")
    dashboard.test_agent(num_tests=10)
    
    print("\n" + "#"*70)
    print("#  Demo complete!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run demo
    demo_full_system()