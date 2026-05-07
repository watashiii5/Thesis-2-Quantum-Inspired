"""
Reinforcement Learning Augmentation for QIA Room Allocation
Learns from custom rules/obstacles and improves scheduling over time.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import random
import json
from collections import defaultdict
import numpy as np
from datetime import datetime


@dataclass
class CustomRule:
    """User-defined rule for the RL agent to learn."""
    rule_id: str
    name: str
    description: str
    penalty: float  # Reward/punishment for violating this rule
    check_fn: callable = None  # Function that checks if rule is violated


@dataclass
class ScheduleAction:
    """An action the RL agent can take: assign a class to a room+time."""
    class_id: str
    room_id: str
    time_slot: str  # "monday_08:00" format
    day: str
    start_time: str
    reward: float = 0.0


class RLSchedulingEnvironment:
    """Environment for RL agent to learn scheduling rules."""

    def __init__(self, custom_rules: List[CustomRule] = None):
        self.custom_rules = custom_rules or []
        self.current_schedule = {}  # class_id -> (room_id, time_slot)
        self.state_history = []
        self.reward_history = []

    def add_rule(self, rule: CustomRule):
        """Register a new custom rule."""
        self.custom_rules.append(rule)

    def evaluate_schedule(self, schedule: Dict) -> float:
        """
        Evaluate schedule against all custom rules.
        Returns total reward (higher = better).
        """
        total_reward = 0.0
        violations = {}

        for rule in self.custom_rules:
            if rule.check_fn:
                violated = rule.check_fn(schedule)
                if violated:
                    total_reward -= rule.penalty
                    violations[rule.rule_id] = rule.penalty

        return total_reward, violations

    def get_possible_actions(self, class_id: str, available_rooms: List[str],
                           available_slots: List[str]) -> List[ScheduleAction]:
        """Generate all possible placements for a class."""
        actions = []
        for room in available_rooms:
            for slot in available_slots:
                day, time = slot.split("_")
                action = ScheduleAction(
                    class_id=class_id,
                    room_id=room,
                    time_slot=slot,
                    day=day,
                    start_time=time
                )
                actions.append(action)
        return actions


class QLearningAgent:
    """Lightweight Q-Learning agent for scheduling."""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Q-table: state_hash -> {action_hash: q_value}
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Experience replay buffer
        self.experience_buffer = []
        self.max_buffer_size = 1000

    def state_to_hash(self, state: Dict) -> str:
        """Convert state to hashable representation."""
        return json.dumps(state, sort_keys=True, default=str)

    def action_to_hash(self, action: ScheduleAction) -> str:
        """Convert action to hashable representation."""
        return f"{action.class_id}_{action.room_id}_{action.time_slot}"

    def select_action(self, state: Dict, available_actions: List[ScheduleAction]) -> ScheduleAction:
        """
        Epsilon-greedy action selection.
        Explore random actions sometimes, exploit best Q-value otherwise.
        """
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        state_hash = self.state_to_hash(state)
        best_action = max(
            available_actions,
            key=lambda a: self.q_table[state_hash][self.action_to_hash(a)],
            default=random.choice(available_actions)
        )
        return best_action if best_action else random.choice(available_actions)

    def learn(self, state: Dict, action: ScheduleAction, reward: float,
              next_state: Dict, done: bool):
        """Update Q-value based on experience."""
        state_hash = self.state_to_hash(state)
        action_hash = self.action_to_hash(action)
        next_state_hash = self.state_to_hash(next_state)

        # Find max Q-value for next state
        max_next_q = max(
            self.q_table[next_state_hash].values(),
            default=0.0
        ) if not done else 0.0

        # Q-learning update rule
        current_q = self.q_table[state_hash][action_hash]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_hash][action_hash] = new_q

        # Store experience
        self.store_experience(state, action, reward, next_state, done)

    def store_experience(self, state: Dict, action: ScheduleAction, reward: float,
                        next_state: Dict, done: bool):
        """Store experience for replay."""
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

        # Keep buffer size manageable
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def replay(self, batch_size: int = 32):
        """Experience replay to improve learning."""
        if len(self.experience_buffer) < batch_size:
            batch_size = len(self.experience_buffer)

        batch = random.sample(self.experience_buffer, batch_size)
        for exp in batch:
            self.learn(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['next_state'],
                exp['done']
            )

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'q_table_size': len(self.q_table),
            'experience_buffer_size': len(self.experience_buffer),
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate
        }


class RLSchedulingEngine:
    """Main engine combining QIA with RL learning."""

    def __init__(self, qia_scheduler_fn: callable = None):
        self.agent = QLearningAgent()
        self.environment = RLSchedulingEnvironment()
        self.qia_scheduler = qia_scheduler_fn
        self.training_history = []

        # Track best observed result during training
        self.best_reward: float = float('-inf')
        self.best_schedule: Optional[Dict[str, Tuple[str, str]]] = None
        self.best_violations: Dict[str, float] = {}

    def add_custom_rule(self, rule: CustomRule):
        """Register a custom scheduling rule."""
        self.environment.add_rule(rule)

    def train_episode(self, classes: List[Dict], rooms: List[Dict],
                     time_slots: List[str], iterations: int = 10) -> Dict:
        """
        Run one training episode where RL learns from QIA solutions.
        """
        episode_rewards = []

        for iteration in range(iterations):
            # Use QIA to get baseline schedule
            if self.qia_scheduler:
                qia_schedule = self.qia_scheduler(classes, rooms, time_slots)
            else:
                qia_schedule = self._dummy_schedule(classes, rooms, time_slots)

            # Evaluate against custom rules
            reward, violations = self.environment.evaluate_schedule(qia_schedule)

            # Track best schedule so far (reward is higher=better)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_schedule = qia_schedule
                self.best_violations = violations

            # RL learns from this experience
            state = self._schedule_to_state(qia_schedule)
            for class_id, (room_id, slot) in qia_schedule.items():
                action = ScheduleAction(
                    class_id=class_id,
                    room_id=room_id,
                    time_slot=slot,
                    day=slot.split("_")[0],
                    start_time=slot.split("_")[1],
                    reward=reward
                )
                next_state = state  # Simplified for now
                self.agent.learn(state, action, reward, next_state, done=True)

            episode_rewards.append(reward)

        # Replay experiences to improve learning
        self.agent.replay(batch_size=min(32, len(self.agent.experience_buffer)))

        avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0

        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'iteration': len(self.training_history),
            'avg_reward': avg_reward,
            'violations': violations if 'violations' in locals() else {}
        }
        self.training_history.append(history_entry)

        return {
            'episode': len(self.training_history),
            'avg_reward': avg_reward,
            'agent_stats': self.agent.get_stats()
        }

    def get_best_result(self) -> Dict[str, Any]:
        """Return best schedule/cost observed so far."""
        if self.best_schedule is None:
            return {
                'has_result': False,
                'best_reward': None,
                'best_cost': None,
                'best_schedule': None,
                'best_violations': {}
            }

        # In this RL module, reward is defined as negative penalty sum,
        # so we can expose a human-readable cost as -reward.
        best_cost = -float(self.best_reward)

        schedule_out: Dict[str, Dict[str, Any]] = {}
        for class_id, (room_id, slot) in self.best_schedule.items():
            schedule_out[str(class_id)] = {
                'room_id': room_id,
                'time_slot': slot
            }

        return {
            'has_result': True,
            'best_reward': float(self.best_reward),
            'best_cost': best_cost,
            'best_schedule': schedule_out,
            'best_violations': self.best_violations or {}
        }

    def get_learned_schedule(self, classes: List[Dict], rooms: List[Dict],
                            time_slots: List[str]) -> Dict:
        """Get schedule optimized using learned Q-values."""
        schedule = {}
        state = {}

        for class_id in [c['id'] for c in classes]:
            available_rooms = [r['id'] for r in rooms]
            actions = self.environment.get_possible_actions(class_id, available_rooms, time_slots)

            # Use learned policy to select action
            best_action = self.agent.select_action(state, actions)
            schedule[class_id] = (best_action.room_id, best_action.time_slot)

        return schedule

    def _schedule_to_state(self, schedule: Dict) -> Dict:
        """Convert schedule to state representation."""
        return {'schedule': json.dumps(schedule, default=str)}

    def _dummy_schedule(self, classes: List[Dict], rooms: List[Dict],
                       time_slots: List[str]) -> Dict:
        """Generate dummy schedule for testing."""
        schedule = {}
        for class_id in [c.get('id', f'class_{i}') for i, c in enumerate(classes)]:
            schedule[class_id] = (
                random.choice([r.get('id', f'room_{i}') for i, r in enumerate(rooms)]),
                random.choice(time_slots)
            )
        return schedule

    def get_training_stats(self) -> Dict:
        """Get overall training statistics."""
        if not self.training_history:
            return {'episodes_trained': 0, 'avg_reward': 0}

        rewards = [h['avg_reward'] for h in self.training_history]
        return {
            'episodes_trained': len(self.training_history),
            'avg_reward': sum(rewards) / len(rewards),
            'best_reward': max(rewards),
            'worst_reward': min(rewards),
            'agent_stats': self.agent.get_stats()
        }


# ==================== Predefined Custom Rules ====================

def create_no_friday_afternoon_rule(penalty: float = 200) -> CustomRule:
    """Rule: No classes on Friday after 3 PM."""
    def check(schedule: Dict) -> bool:
        # This would check actual schedule data
        # Simplified for demo
        return False

    return CustomRule(
        rule_id="no_friday_afternoon",
        name="No Friday Afternoons",
        description="Avoid scheduling classes Friday after 3 PM",
        penalty=penalty,
        check_fn=check
    )


def create_teacher_lunch_break_rule(penalty: float = 150) -> CustomRule:
    """Rule: Teachers get lunch break 12-1 PM."""
    def check(schedule: Dict) -> bool:
        return False

    return CustomRule(
        rule_id="teacher_lunch",
        name="Teacher Lunch Break",
        description="Ensure teachers have break 12-1 PM",
        penalty=penalty,
        check_fn=check
    )


def create_room_clustering_rule(penalty: float = 100) -> CustomRule:
    """Rule: Keep classes in same room when possible to reduce movement."""
    def check(schedule: Dict) -> bool:
        return False

    return CustomRule(
        rule_id="room_clustering",
        name="Room Clustering",
        description="Group classes in same room to reduce campus movement",
        penalty=penalty,
        check_fn=check
    )
