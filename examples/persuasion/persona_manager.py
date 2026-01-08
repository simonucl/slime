"""
PersonaManager for loading and managing Persuasion for Good dataset.

This module handles loading ConvoKit corpus data (conversations, users, utterances)
and provides task lookup and persona description generation.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PersonaAttributes:
    """Persona attributes from ConvoKit users.json"""
    user_id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    race: Optional[str] = None
    edu: Optional[str] = None
    income: Optional[float] = None
    employment: Optional[str] = None
    marital: Optional[str] = None
    religion: Optional[str] = None
    ideology: Optional[str] = None

    # Big Five personality traits
    extrovert: Optional[float] = None
    agreeable: Optional[float] = None
    conscientious: Optional[float] = None
    neurotic: Optional[float] = None
    open: Optional[float] = None

    # Schwartz values
    achievement: Optional[float] = None
    benevolence: Optional[float] = None
    conform: Optional[float] = None
    hedonism: Optional[float] = None
    power: Optional[float] = None
    security: Optional[float] = None
    self_direction: Optional[float] = None
    stimulation: Optional[float] = None
    tradition: Optional[float] = None
    universalism: Optional[float] = None

    # Moral Foundations Theory
    care: Optional[float] = None
    fairness: Optional[float] = None
    loyalty: Optional[float] = None
    authority: Optional[float] = None
    purity: Optional[float] = None

    # Decision-making style
    rational: Optional[float] = None
    intuitive: Optional[float] = None

    # Liberty Foundation
    freedom: Optional[float] = None


@dataclass
class PersuasionTask:
    """Represents one persuasion conversation task"""
    conversation_id: str
    persuader_id: str
    persuadee_id: str
    split: str  # "train" or "test"
    persuader_persona: PersonaAttributes
    persuadee_persona: PersonaAttributes
    ground_truth_donation: Optional[float] = None  # For eval only
    dialogue_id: Optional[str] = None  # Original dialogue ID from corpus


class PersonaManager:
    """Manages ConvoKit corpus and persona descriptions for Persuasion for Good dataset"""

    def __init__(self, corpus_path: str):
        """
        Initialize PersonaManager and load ConvoKit corpus.

        Args:
            corpus_path: Path to persuasionforgood_corpus directory containing
                        conversations.json, users.json, and utterances.jsonl
        """
        self.corpus_path = corpus_path

        # Load corpus data
        self.conversations = self._load_conversations()
        self.users = self._load_users()
        self.utterances = self._load_utterances()

        # Build task lookup
        self.tasks = self._build_tasks()

    def _load_conversations(self) -> dict:
        """Load conversations.json"""
        path = os.path.join(self.corpus_path, "conversations.json")
        with open(path, 'r') as f:
            return json.load(f)

    def _load_users(self) -> dict:
        """Load users.json and convert to PersonaAttributes"""
        path = os.path.join(self.corpus_path, "users.json")
        with open(path, 'r') as f:
            users_data = json.load(f)

        # Convert to PersonaAttributes
        users = {}
        for user_id, attrs in users_data.items():
            users[user_id] = PersonaAttributes(
                user_id=user_id,
                **{k: v for k, v in attrs.items() if k in PersonaAttributes.__dataclass_fields__}
            )
        return users

    def _load_utterances(self) -> dict:
        """Load utterances.jsonl and organize by conversation ID"""
        path = os.path.join(self.corpus_path, "utterances.jsonl")
        utterances_by_conv = {}

        with open(path, 'r') as f:
            for line in f:
                utt = json.loads(line)
                conv_id = utt['root']
                if conv_id not in utterances_by_conv:
                    utterances_by_conv[conv_id] = []
                utterances_by_conv[conv_id].append(utt)

        # Sort utterances by turn ID within each conversation
        for conv_id in utterances_by_conv:
            utterances_by_conv[conv_id].sort(
                key=lambda u: u['meta']['user_turn_id']
            )

        return utterances_by_conv

    def _build_tasks(self) -> dict:
        """Build PersuasionTask objects for all conversations"""
        tasks = {}

        for conv_id, conv_data in self.conversations.items():
            persuader_id = conv_data['user_er']
            persuadee_id = conv_data['user_ee']

            # Determine split based on is_annotated
            # Following the pattern from persuasion_simulation:
            # Annotated conversations are typically used for test/evaluation
            split = "test" if conv_data.get('is_annotated', False) else "train"

            # Get personas
            persuader_persona = self.users.get(persuader_id)
            persuadee_persona = self.users.get(persuadee_id)

            if persuader_persona is None or persuadee_persona is None:
                # Skip conversations with missing user data
                continue

            # Get ground truth donation (persuadee's donation)
            donation = conv_data.get('donation_ee')
            if donation is not None and str(donation).lower() != 'nan':
                ground_truth_donation = float(donation)
            else:
                ground_truth_donation = None

            tasks[conv_id] = PersuasionTask(
                conversation_id=conv_id,
                persuader_id=persuader_id,
                persuadee_id=persuadee_id,
                split=split,
                persuader_persona=persuader_persona,
                persuadee_persona=persuadee_persona,
                ground_truth_donation=ground_truth_donation,
                dialogue_id=conv_data.get('dialogue_id')
            )

        return tasks

    def get_task(self, conversation_id: str) -> Optional[PersuasionTask]:
        """
        Get task by conversation ID.

        Args:
            conversation_id: Conversation ID (e.g., "0", "21", "41")

        Returns:
            PersuasionTask or None if not found
        """
        return self.tasks.get(conversation_id)

    def get_tasks_by_split(self, split: str) -> list[PersuasionTask]:
        """
        Get all tasks for a specific split.

        Args:
            split: "train" or "test"

        Returns:
            List of PersuasionTask objects
        """
        return [task for task in self.tasks.values() if task.split == split]

    def get_conversation_utterances(self, conversation_id: str) -> list[dict]:
        """
        Get all utterances for a conversation, sorted by turn order.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of utterance dicts
        """
        return self.utterances.get(conversation_id, [])

    def _is_valid_number(self, value) -> bool:
        """Check if a value is a valid number (not None and not NaN)"""
        return value is not None and not (isinstance(value, float) and math.isnan(value))

    def generate_persona_description(self, attrs: PersonaAttributes) -> str:
        """
        Generate natural language persona description from attributes.

        This follows the pattern from persuasion_simulation/src/prompts/persona.py

        Args:
            attrs: PersonaAttributes object

        Returns:
            Natural language persona description string
        """
        lines = []

        # Demographics
        if self._is_valid_number(attrs.age):
            lines.append(f"Age: {int(attrs.age)}")
        if attrs.sex:
            lines.append(f"Gender: {attrs.sex}")
        if attrs.race:
            lines.append(f"Race: {attrs.race}")
        if attrs.edu:
            lines.append(f"Education: {attrs.edu}")
        if self._is_valid_number(attrs.income):
            income_map = {
                1.0: "Less than $10,000",
                2.0: "$10,000 - $24,999",
                3.0: "$25,000 - $49,999",
                4.0: "$50,000 - $74,999",
                5.0: "$75,000 - $99,999",
                6.0: "$100,000 - $149,999",
                7.0: "$150,000 or more"
            }
            income_str = income_map.get(attrs.income, f"Income level {attrs.income}")
            lines.append(f"Income: {income_str}")
        if attrs.employment:
            lines.append(f"Employment: {attrs.employment}")
        if attrs.marital:
            lines.append(f"Marital Status: {attrs.marital}")
        if attrs.religion:
            lines.append(f"Religion: {attrs.religion}")
        if attrs.ideology:
            lines.append(f"Political Ideology: {attrs.ideology}")

        # Personality (Big Five) - only include if available
        personality_traits = []
        if self._is_valid_number(attrs.extrovert):
            personality_traits.append(f"extraversion: {attrs.extrovert:.1f}/5")
        if self._is_valid_number(attrs.agreeable):
            personality_traits.append(f"agreeableness: {attrs.agreeable:.1f}/5")
        if self._is_valid_number(attrs.conscientious):
            personality_traits.append(f"conscientiousness: {attrs.conscientious:.1f}/5")
        if self._is_valid_number(attrs.neurotic):
            personality_traits.append(f"neuroticism: {attrs.neurotic:.1f}/5")
        if self._is_valid_number(attrs.open):
            personality_traits.append(f"openness: {attrs.open:.1f}/5")

        if personality_traits:
            lines.append(f"Personality traits: {', '.join(personality_traits)}")

        # Values (Schwartz) - only include top values
        value_scores = {}
        if self._is_valid_number(attrs.achievement):
            value_scores['achievement'] = attrs.achievement
        if self._is_valid_number(attrs.benevolence):
            value_scores['benevolence'] = attrs.benevolence
        if self._is_valid_number(attrs.power):
            value_scores['power'] = attrs.power
        if self._is_valid_number(attrs.security):
            value_scores['security'] = attrs.security
        if self._is_valid_number(attrs.self_direction):
            value_scores['self-direction'] = attrs.self_direction
        if self._is_valid_number(attrs.universalism):
            value_scores['universalism'] = attrs.universalism

        if value_scores:
            # Get top 3 values
            sorted_values = sorted(value_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            values_str = ', '.join([f"{v[0]} ({v[1]:.1f})" for v in sorted_values])
            lines.append(f"Core values: {values_str}")

        # Moral foundations - only include strong ones (> 3.5)
        moral_foundations = []
        if self._is_valid_number(attrs.care) and attrs.care > 3.5:
            moral_foundations.append(f"care: {attrs.care:.1f}")
        if self._is_valid_number(attrs.fairness) and attrs.fairness > 3.5:
            moral_foundations.append(f"fairness: {attrs.fairness:.1f}")
        if self._is_valid_number(attrs.loyalty) and attrs.loyalty > 3.5:
            moral_foundations.append(f"loyalty: {attrs.loyalty:.1f}")
        if self._is_valid_number(attrs.authority) and attrs.authority > 3.5:
            moral_foundations.append(f"authority: {attrs.authority:.1f}")

        if moral_foundations:
            lines.append(f"Strong moral foundations: {', '.join(moral_foundations)}")

        # Decision-making style
        if self._is_valid_number(attrs.rational) and self._is_valid_number(attrs.intuitive):
            if attrs.rational > attrs.intuitive:
                lines.append(f"Decision-making style: More rational (rational: {attrs.rational:.1f}, intuitive: {attrs.intuitive:.1f})")
            elif attrs.intuitive > attrs.rational:
                lines.append(f"Decision-making style: More intuitive (rational: {attrs.rational:.1f}, intuitive: {attrs.intuitive:.1f})")
            else:
                lines.append(f"Decision-making style: Balanced (rational: {attrs.rational:.1f}, intuitive: {attrs.intuitive:.1f})")

        return "\n".join(lines)
