"""
NoiseInjector: Context-aware distractor injection for GSM-NoOp dataset.

Implements the "seductive detail effect" by injecting topically relevant
but mathematically irrelevant sentences into math problems.

Reference: Phase 2 Specification Task 2.2.2 and Section 5
"""

import re
import random
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class TopicDomain(Enum):
    """Topic domains for distractor categorization."""
    AGRICULTURE = "agriculture"
    COMMERCE = "commerce"
    INVENTORY = "inventory"
    TRAVEL = "travel"
    TIME = "time"
    NATURE = "nature"
    EDUCATION = "education"
    CONSTRUCTION = "construction"
    GENERAL = "general"


@dataclass
class DistractorTemplate:
    """A distractor sentence template with metadata."""
    template: str
    domain: TopicDomain
    requires_number: bool = True
    requires_object: bool = True
    number_constraint: str = "smaller"  # 'smaller', 'any', 'time', 'percentage'


# Comprehensive Topical Distractor Library
# Templates use {N} for numbers and {object} for relevant nouns
DISTRACTOR_LIBRARY: Dict[TopicDomain, List[DistractorTemplate]] = {
    TopicDomain.AGRICULTURE: [
        # Quality/State Attributes - these are "seemingly relevant" but don't affect count
        DistractorTemplate(
            "Note that {N} of the {object} were slightly bruised.",
            TopicDomain.AGRICULTURE, True, True, "smaller"
        ),
        DistractorTemplate(
            "However, {N} of the {object} were unripe.",
            TopicDomain.AGRICULTURE, True, True, "smaller"
        ),
        DistractorTemplate(
            "Interestingly, {N} of the {object} were smaller than average.",
            TopicDomain.AGRICULTURE, True, True, "smaller"
        ),
        DistractorTemplate(
            "It's worth noting that {N} of the {object} were organically grown.",
            TopicDomain.AGRICULTURE, True, True, "smaller"
        ),
        DistractorTemplate(
            "About {N} of the {object} had minor blemishes.",
            TopicDomain.AGRICULTURE, True, True, "smaller"
        ),
        DistractorTemplate(
            "{object} are known to be rich in vitamins and nutrients.",
            TopicDomain.AGRICULTURE, False, True, "any"
        ),
        DistractorTemplate(
            "The {object} were freshly picked that morning.",
            TopicDomain.AGRICULTURE, False, True, "any"
        ),
        DistractorTemplate(
            "This variety of {object} is particularly popular in summer.",
            TopicDomain.AGRICULTURE, False, True, "any"
        ),
        DistractorTemplate(
            "{N} of the {object} were from a local farm.",
            TopicDomain.AGRICULTURE, True, True, "smaller"
        ),
        DistractorTemplate(
            "The farmer mentioned that {N} of the {object} were premium grade.",
            TopicDomain.AGRICULTURE, True, True, "smaller"
        ),
    ],
    TopicDomain.COMMERCE: [
        # Temporal Pricing / Historical context - numbers that don't affect current calculation
        DistractorTemplate(
            "The price of {object} was ${N} lower last week.",
            TopicDomain.COMMERCE, True, True, "any"
        ),
        DistractorTemplate(
            "Last month, {object} cost {N}% more than today.",
            TopicDomain.COMMERCE, True, True, "percentage"
        ),
        DistractorTemplate(
            "They had originally considered buying {N} different items.",
            TopicDomain.COMMERCE, True, False, "any"
        ),
        DistractorTemplate(
            "The store had a promotion last week offering {N}% off.",
            TopicDomain.COMMERCE, True, False, "percentage"
        ),
        DistractorTemplate(
            "The receipt showed {N} loyalty points were earned.",
            TopicDomain.COMMERCE, True, False, "any"
        ),
        DistractorTemplate(
            "The store closes at {N} PM on weekdays.",
            TopicDomain.COMMERCE, True, False, "time"
        ),
        DistractorTemplate(
            "There were {N} other customers in the store at the time.",
            TopicDomain.COMMERCE, True, False, "any"
        ),
        DistractorTemplate(
            "The cashier mentioned that {N} people had bought the same item earlier.",
            TopicDomain.COMMERCE, True, False, "any"
        ),
        DistractorTemplate(
            "The store has been in business for {N} years.",
            TopicDomain.COMMERCE, True, False, "any"
        ),
        DistractorTemplate(
            "Shipping would have cost an additional ${N} for delivery.",
            TopicDomain.COMMERCE, True, False, "any"
        ),
    ],
    TopicDomain.INVENTORY: [
        # Physical Properties - attributes that don't change quantities
        DistractorTemplate(
            "{N} of the {object} were old and slightly worn.",
            TopicDomain.INVENTORY, True, True, "smaller"
        ),
        DistractorTemplate(
            "{N} of the {object} were gifts from a friend.",
            TopicDomain.INVENTORY, True, True, "smaller"
        ),
        DistractorTemplate(
            "Notably, {N} of the {object} were collector's editions.",
            TopicDomain.INVENTORY, True, True, "smaller"
        ),
        DistractorTemplate(
            "{N} of the {object} were still in their original packaging.",
            TopicDomain.INVENTORY, True, True, "smaller"
        ),
        DistractorTemplate(
            "The {object} were arranged neatly on {N} different shelves.",
            TopicDomain.INVENTORY, True, True, "any"
        ),
        DistractorTemplate(
            "Each {object} weighs approximately {N} ounces.",
            TopicDomain.INVENTORY, True, True, "any"
        ),
        DistractorTemplate(
            "The collection started {N} years ago.",
            TopicDomain.INVENTORY, True, True, "any"
        ),
        DistractorTemplate(
            "{N} of the {object} were duplicates.",
            TopicDomain.INVENTORY, True, True, "smaller"
        ),
        DistractorTemplate(
            "The storage box could hold up to {N} more {object}.",
            TopicDomain.INVENTORY, True, True, "any"
        ),
        DistractorTemplate(
            "{N} of the {object} were particularly rare.",
            TopicDomain.INVENTORY, True, True, "smaller"
        ),
    ],
    TopicDomain.TRAVEL: [
        # Weather/Stops - contextual information that doesn't affect distance/time calculations
        DistractorTemplate(
            "It started raining {N} minutes into the trip.",
            TopicDomain.TRAVEL, True, False, "any"
        ),
        DistractorTemplate(
            "The weather forecast predicted {N}% chance of rain.",
            TopicDomain.TRAVEL, True, False, "percentage"
        ),
        DistractorTemplate(
            "The driver had made this trip {N} times before.",
            TopicDomain.TRAVEL, True, False, "any"
        ),
        DistractorTemplate(
            "There were {N} gas stations along the route.",
            TopicDomain.TRAVEL, True, False, "any"
        ),
        DistractorTemplate(
            "The road had {N} traffic lights in total.",
            TopicDomain.TRAVEL, True, False, "any"
        ),
        DistractorTemplate(
            "The GPS showed {N} alternate routes available.",
            TopicDomain.TRAVEL, True, False, "any"
        ),
        DistractorTemplate(
            "The car's fuel tank could hold {N} gallons.",
            TopicDomain.TRAVEL, True, False, "any"
        ),
        DistractorTemplate(
            "The temperature outside was {N} degrees.",
            TopicDomain.TRAVEL, True, False, "any"
        ),
        DistractorTemplate(
            "The speedometer showed the car had traveled {N} miles in its lifetime.",
            TopicDomain.TRAVEL, True, False, "any"
        ),
        DistractorTemplate(
            "They passed {N} scenic viewpoints along the way.",
            TopicDomain.TRAVEL, True, False, "any"
        ),
    ],
    TopicDomain.TIME: [
        # Calendar Trivia - temporal facts that don't affect calculations
        DistractorTemplate(
            "The previous month had {N} days.",
            TopicDomain.TIME, True, False, "any"
        ),
        DistractorTemplate(
            "It was week {N} of the year.",
            TopicDomain.TIME, True, False, "any"
        ),
        DistractorTemplate(
            "The calendar showed {N} holidays that month.",
            TopicDomain.TIME, True, False, "any"
        ),
        DistractorTemplate(
            "There were {N} weekends left in the month.",
            TopicDomain.TIME, True, False, "any"
        ),
        DistractorTemplate(
            "The day had {N} hours of daylight.",
            TopicDomain.TIME, True, False, "any"
        ),
        DistractorTemplate(
            "It was the {N}th day of the month.",
            TopicDomain.TIME, True, False, "any"
        ),
        DistractorTemplate(
            "The timezone was {N} hours ahead of UTC.",
            TopicDomain.TIME, True, False, "any"
        ),
        DistractorTemplate(
            "Sunrise was at {N} AM that day.",
            TopicDomain.TIME, True, False, "time"
        ),
        DistractorTemplate(
            "The schedule had been updated {N} times this week.",
            TopicDomain.TIME, True, False, "any"
        ),
        DistractorTemplate(
            "There were {N} appointments scheduled for later that day.",
            TopicDomain.TIME, True, False, "any"
        ),
    ],
    TopicDomain.NATURE: [
        DistractorTemplate(
            "The weather had been mild for {N} consecutive days.",
            TopicDomain.NATURE, True, False, "any"
        ),
        DistractorTemplate(
            "Scientists estimate this phenomenon occurs every {N} years.",
            TopicDomain.NATURE, True, False, "any"
        ),
        DistractorTemplate(
            "The area typically receives {N} inches of rainfall annually.",
            TopicDomain.NATURE, True, False, "any"
        ),
        DistractorTemplate(
            "Visibility was approximately {N} miles that day.",
            TopicDomain.NATURE, True, False, "any"
        ),
        DistractorTemplate(
            "The local wildlife sanctuary houses over {N} species.",
            TopicDomain.NATURE, True, False, "any"
        ),
        DistractorTemplate(
            "Wind speeds reached {N} miles per hour at the peak.",
            TopicDomain.NATURE, True, False, "any"
        ),
        DistractorTemplate(
            "The water temperature was {N} degrees.",
            TopicDomain.NATURE, True, False, "any"
        ),
        DistractorTemplate(
            "The ecosystem supports approximately {N} different plant species.",
            TopicDomain.NATURE, True, False, "any"
        ),
    ],
    TopicDomain.EDUCATION: [
        DistractorTemplate(
            "The class had {N} students enrolled last semester.",
            TopicDomain.EDUCATION, True, False, "any"
        ),
        DistractorTemplate(
            "The textbook had {N} chapters in total.",
            TopicDomain.EDUCATION, True, False, "any"
        ),
        DistractorTemplate(
            "The library contained over {N} books on the subject.",
            TopicDomain.EDUCATION, True, False, "any"
        ),
        DistractorTemplate(
            "The average test score for the class was {N}%.",
            TopicDomain.EDUCATION, True, False, "percentage"
        ),
        DistractorTemplate(
            "The school had been established {N} years ago.",
            TopicDomain.EDUCATION, True, False, "any"
        ),
        DistractorTemplate(
            "There were {N} teachers in the department.",
            TopicDomain.EDUCATION, True, False, "any"
        ),
        DistractorTemplate(
            "The homework assignment had {N} problems in total.",
            TopicDomain.EDUCATION, True, False, "any"
        ),
        DistractorTemplate(
            "The semester lasted {N} weeks.",
            TopicDomain.EDUCATION, True, False, "any"
        ),
    ],
    TopicDomain.CONSTRUCTION: [
        DistractorTemplate(
            "The building was constructed {N} years ago.",
            TopicDomain.CONSTRUCTION, True, False, "any"
        ),
        DistractorTemplate(
            "The property had {N} trees in the yard.",
            TopicDomain.CONSTRUCTION, True, False, "any"
        ),
        DistractorTemplate(
            "The house had {N} windows on the ground floor.",
            TopicDomain.CONSTRUCTION, True, False, "any"
        ),
        DistractorTemplate(
            "The renovation project took {N} months to complete.",
            TopicDomain.CONSTRUCTION, True, False, "any"
        ),
        DistractorTemplate(
            "The neighborhood had {N} similar houses.",
            TopicDomain.CONSTRUCTION, True, False, "any"
        ),
        DistractorTemplate(
            "The property value had increased by {N}% over the past year.",
            TopicDomain.CONSTRUCTION, True, False, "percentage"
        ),
        DistractorTemplate(
            "The ceiling height was {N} feet.",
            TopicDomain.CONSTRUCTION, True, False, "any"
        ),
        DistractorTemplate(
            "There were {N} light fixtures in the room.",
            TopicDomain.CONSTRUCTION, True, False, "any"
        ),
    ],
    TopicDomain.GENERAL: [
        # Generic distractors that can apply to most contexts
        DistractorTemplate(
            "Interestingly, {N} people had faced a similar situation before.",
            TopicDomain.GENERAL, True, False, "any"
        ),
        DistractorTemplate(
            "This type of problem typically takes {N} minutes to solve.",
            TopicDomain.GENERAL, True, False, "any"
        ),
        DistractorTemplate(
            "According to recent surveys, {N}% of people prefer this approach.",
            TopicDomain.GENERAL, True, False, "percentage"
        ),
        DistractorTemplate(
            "The average person encounters this scenario {N} times a year.",
            TopicDomain.GENERAL, True, False, "any"
        ),
        DistractorTemplate(
            "Historical records show this has happened {N} times before.",
            TopicDomain.GENERAL, True, False, "any"
        ),
        DistractorTemplate(
            "Experts recommend reviewing this every {N} months.",
            TopicDomain.GENERAL, True, False, "any"
        ),
        DistractorTemplate(
            "The procedure has {N} optional steps that can be skipped.",
            TopicDomain.GENERAL, True, False, "any"
        ),
        DistractorTemplate(
            "Approximately {N} alternatives were considered.",
            TopicDomain.GENERAL, True, False, "any"
        ),
    ],
}


@dataclass
class InjectedProblem:
    """A problem with injected distractor(s)."""
    original_question: str
    modified_question: str
    distractors_used: List[str]
    injection_positions: List[str]  # 'start', 'middle', 'end'
    answer_unchanged: bool


class NoiseInjector:
    """
    Injects context-aware distractors into math problems.

    Creates GSM-NoOp variants by adding "seemingly relevant" but
    mathematically irrelevant sentences to trigger model failures.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the NoiseInjector.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self._topic_mapping = self._build_topic_mapping()

    def _build_topic_mapping(self) -> Dict[str, TopicDomain]:
        """Build keyword to topic mapping for auto-detection."""
        mapping = {}
        keyword_topics = {
            TopicDomain.AGRICULTURE: [
                'eat', 'bake', 'fruit', 'cook', 'slice', 'apple', 'orange',
                'vegetable', 'farm', 'harvest', 'plant', 'grow', 'food',
                'banana', 'tomato', 'potato', 'carrot', 'lettuce', 'berry',
                'grape', 'melon', 'peach', 'pear', 'cherry', 'lemon', 'kiwi'
            ],
            TopicDomain.COMMERCE: [
                'buy', 'sell', 'cost', 'price', 'dollar', 'pay', 'money',
                'store', 'shop', 'discount', 'spend', 'earn', 'profit',
                'purchase', 'sale', 'market', 'customer', 'cashier', 'receipt'
            ],
            TopicDomain.INVENTORY: [
                'book', 'toy', 'card', 'stamp', 'collect', 'item', 'box',
                'container', 'stack', 'pile', 'shelf', 'marble', 'sticker',
                'comic', 'coin', 'figurine', 'doll', 'game', 'puzzle'
            ],
            TopicDomain.TRAVEL: [
                'drive', 'mile', 'speed', 'trip', 'travel', 'distance',
                'car', 'train', 'plane', 'walk', 'run', 'bike', 'road',
                'highway', 'route', 'destination', 'journey', 'commute'
            ],
            TopicDomain.TIME: [
                'day', 'week', 'month', 'hour', 'minute', 'year', 'schedule',
                'calendar', 'morning', 'evening', 'night', 'afternoon',
                'deadline', 'appointment', 'shift', 'period'
            ],
            TopicDomain.NATURE: [
                'fog', 'rain', 'weather', 'ocean', 'river', 'lake', 'mountain',
                'forest', 'animal', 'fish', 'bird', 'tree', 'flower', 'storm',
                'wind', 'cloud', 'sun', 'snow', 'desert', 'beach'
            ],
            TopicDomain.EDUCATION: [
                'student', 'teacher', 'class', 'school', 'grade', 'exam',
                'homework', 'study', 'learn', 'read', 'write', 'test',
                'lecture', 'professor', 'university', 'college', 'course'
            ],
            TopicDomain.CONSTRUCTION: [
                'build', 'house', 'room', 'wall', 'floor', 'area',
                'perimeter', 'length', 'width', 'height', 'square',
                'feet', 'meter', 'yard', 'inch', 'building', 'apartment'
            ],
        }
        for topic, keywords in keyword_topics.items():
            for kw in keywords:
                mapping[kw] = topic
        return mapping

    def detect_topics(self, text: str) -> Set[TopicDomain]:
        """Detect topic domains from text content."""
        text_lower = text.lower()
        topics = set()

        for keyword, topic in self._topic_mapping.items():
            if keyword in text_lower:
                topics.add(topic)

        return topics if topics else {TopicDomain.GENERAL}

    def extract_objects(self, text: str) -> List[str]:
        """Extract noun objects from the problem text."""
        # Common object patterns in math problems
        object_patterns = [
            r'(\d+)\s+(\w+(?:s|es)?)\b',  # "5 apples"
            r'the\s+(\w+(?:s|es)?)\b',     # "the apples"
            r'some\s+(\w+(?:s|es)?)\b',    # "some books"
            r'many\s+(\w+(?:s|es)?)\b',    # "many toys"
        ]

        objects = set()
        for pattern in object_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                obj = match[1] if isinstance(match, tuple) else match
                # Filter out common non-objects
                if obj not in ['the', 'a', 'an', 'is', 'are', 'was', 'were',
                              'minutes', 'hours', 'days', 'dollars', 'feet',
                              'miles', 'percent', 'times']:
                    objects.add(obj)

        return list(objects) if objects else ['items']

    def extract_quantities(self, text: str) -> List[int]:
        """Extract numeric quantities from the problem text."""
        numbers = re.findall(r'\b(\d+)\b', text)
        return [int(n) for n in numbers if int(n) > 0]

    def generate_distractor_number(self, constraint: str, max_quantity: int) -> int:
        """Generate an appropriate number for the distractor."""
        if constraint == "smaller":
            # Must be smaller than the main quantity to be plausible
            upper = max(2, min(max_quantity - 1, max_quantity // 2))
            return self.rng.randint(1, upper)
        elif constraint == "percentage":
            return self.rng.randint(5, 95)
        elif constraint == "time":
            return self.rng.randint(1, 12)
        else:  # 'any'
            return self.rng.randint(2, 50)

    def select_distractor(self, topics: Set[TopicDomain],
                          objects: List[str],
                          max_quantity: int) -> Tuple[str, DistractorTemplate]:
        """Select and populate an appropriate distractor."""
        # Prefer topic-specific distractors
        available_templates = []
        for topic in topics:
            if topic in DISTRACTOR_LIBRARY:
                available_templates.extend(DISTRACTOR_LIBRARY[topic])

        # Fall back to general if no topic-specific available
        if not available_templates:
            available_templates = DISTRACTOR_LIBRARY[TopicDomain.GENERAL]

        template = self.rng.choice(available_templates)

        # Generate the distractor text
        distractor_text = template.template

        # Replace {N} with appropriate number
        if template.requires_number and '{N}' in distractor_text:
            num = self.generate_distractor_number(template.number_constraint, max_quantity)
            distractor_text = distractor_text.replace('{N}', str(num))

        # Replace {object} with relevant object
        if template.requires_object and '{object}' in distractor_text:
            obj = self.rng.choice(objects) if objects else 'items'
            distractor_text = distractor_text.replace('{object}', obj)

        return distractor_text, template

    def find_injection_point(self, text: str, position: str) -> int:
        """Find the character index for injection based on position."""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if position == 'start':
            # After the first sentence
            if len(sentences) > 1:
                return len(sentences[0]) + 1
            return 0

        elif position == 'end':
            # Before the question (usually the last sentence starting with question words)
            for _, sent in enumerate(reversed(sentences)):
                if any(sent.lower().startswith(q) for q in
                       ['how', 'what', 'when', 'where', 'who', 'which', 'if']):
                    # Insert before this question
                    pos = len(text) - len(sent)
                    return max(0, pos - 1)
            # Default: before last sentence
            if len(sentences) > 1:
                return len(text) - len(sentences[-1])
            return len(text)

        else:  # 'middle'
            # After approximately half the sentences
            if len(sentences) > 2:
                mid_idx = len(sentences) // 2
                pos = sum(len(s) + 1 for s in sentences[:mid_idx])
                return pos
            return len(text) // 2

    def inject_distractors(self, question: str, num_distractors: int = 1,
                           topics: Optional[Set[TopicDomain]] = None) -> InjectedProblem:
        """
        Inject distractor sentences into a question.

        Args:
            question: Original question text
            num_distractors: Number of distractors to inject (1-3)
            topics: Pre-detected topics (optional)

        Returns:
            InjectedProblem with modified question
        """
        # Clamp number of distractors
        num_distractors = max(1, min(3, num_distractors))

        # Detect topics if not provided
        if topics is None:
            topics = self.detect_topics(question)

        # Extract context
        objects = self.extract_objects(question)
        quantities = self.extract_quantities(question)
        max_quantity = max(quantities) if quantities else 10

        # Generate distractors
        distractors = []
        positions = ['middle', 'start', 'end'][:num_distractors]
        self.rng.shuffle(positions)

        modified_question = question

        for position in positions:
            distractor_text, _ = self.select_distractor(topics, objects, max_quantity)
            distractors.append(distractor_text)

            # Find injection point and insert
            idx = self.find_injection_point(modified_question, position)
            modified_question = (
                modified_question[:idx].rstrip() + ' ' +
                distractor_text + ' ' +
                modified_question[idx:].lstrip()
            )

        return InjectedProblem(
            original_question=question,
            modified_question=modified_question.strip(),
            distractors_used=distractors,
            injection_positions=positions,
            answer_unchanged=True  # By design, distractors don't change the answer
        )

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = random.Random(seed)


if __name__ == "__main__":
    # Demo usage
    injector = NoiseInjector(seed=42)

    # Test problem
    test_question = """Janet has 10 apples. She gives 3 apples to her friend
    and then buys 5 more apples from the store. How many apples does Janet have now?"""

    print("Original question:")
    print(test_question)
    print()

    # Inject 1 distractor
    result = injector.inject_distractors(test_question, num_distractors=1)
    print("With 1 distractor:")
    print(result.modified_question)
    print(f"Distractors: {result.distractors_used}")
    print()

    # Inject 2 distractors
    result = injector.inject_distractors(test_question, num_distractors=2)
    print("With 2 distractors:")
    print(result.modified_question)
    print(f"Distractors: {result.distractors_used}")
    print()

    # Test topic detection
    print("Detected topics:", injector.detect_topics(test_question))
    print("Extracted objects:", injector.extract_objects(test_question))
