"""
TemplateEngine: Core parser and generator for GSM-Symbolic templates.

Parses JSON templates with variable placeholders, evaluates conditions,
and generates diverse problem instances with mathematically correct answers.

Reference: Phase 2 Specification Task 2.2.1
"""

import re
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from fractions import Fraction

from .name_provider import NameProvider


# Built-in sample collections used by GSM-Symbolic templates
BUILTIN_COLLECTIONS = {
    'fractions': {
        'half': 0.5,
        'one-third': 1/3,
        'two-thirds': 2/3,
        'one-fourth': 0.25,
        'three-fourths': 0.75,
        'one-fifth': 0.2,
        'two-fifths': 0.4,
        'three-fifths': 0.6,
        'four-fifths': 0.8,
    },
    'fraction_decimals': [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9],
    'currencies_sym': ['$', '€', '£', '¥'],
    'currencies_word': ['dollars', 'euros', 'pounds', 'yen'],
    'time_units': ['minutes', 'hours', 'days', 'weeks'],
    'distance_units': ['miles', 'kilometers', 'meters', 'feet'],
    'weight_units': ['pounds', 'kilograms', 'grams', 'ounces'],
}


@dataclass
class VariableSpec:
    """Specification for a template variable."""
    name: str
    var_type: str  # 'range', 'sample', 'literal'
    params: Any
    is_numeric: bool = False


@dataclass
class TemplateSpec:
    """Parsed template specification."""
    template_id: str
    question_template: str
    answer_template: str
    variables: Dict[str, VariableSpec]
    conditions: List[str]
    answer_formula: str
    original_question: str
    original_answer: str
    original_id: int

    # Metadata for topic detection (used by NoiseInjector)
    detected_topics: Set[str] = field(default_factory=set)


@dataclass
class GeneratedProblem:
    """A generated problem instance."""
    id: str
    template_id: str
    instance: int
    question: str
    answer: str
    final_answer: Any
    variables: Dict[str, Any]
    is_noop: bool = False
    original_question: str = ""
    original_answer: str = ""


class TemplateParser:
    """Parses GSM-Symbolic template format."""

    # Regex patterns for parsing
    VARIABLE_PATTERN = re.compile(r'\{([^}]+)\}')
    INIT_PATTERN = re.compile(r'#init:\s*(.*?)(?=#conditions:|#answer:|$)', re.DOTALL)
    CONDITIONS_PATTERN = re.compile(r'#conditions:\s*(.*?)(?=#answer:|$)', re.DOTALL)
    ANSWER_PATTERN = re.compile(r'#answer:\s*(.+?)$', re.DOTALL)
    RANGE_PATTERN = re.compile(r'range\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?\s*\)')
    SAMPLE_PATTERN = re.compile(r'sample\s*\(\s*(.+?)\s*\)')

    @classmethod
    def parse_template_file(cls, filepath: Path) -> TemplateSpec:
        """Parse a GSM-Symbolic template JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        template_id = filepath.stem
        question_annotated = data.get('question_annotated', '')
        answer_annotated = data.get('answer_annotated', '')

        # Split question from annotations
        parts = question_annotated.split('\n\n#init:')
        question_template = parts[0].strip()
        annotation_block = '#init:' + parts[1] if len(parts) > 1 else ''

        # Parse variables from question template
        variables = cls._parse_variables(question_template)

        # Parse init section
        init_match = cls.INIT_PATTERN.search(annotation_block)
        if init_match:
            init_block = init_match.group(1)
            cls._parse_init_block(init_block, variables)

        # Parse conditions
        conditions = []
        cond_match = cls.CONDITIONS_PATTERN.search(annotation_block)
        if cond_match:
            cond_block = cond_match.group(1)
            conditions = cls._parse_conditions(cond_block)

        # Parse answer formula
        answer_formula = ""
        ans_match = cls.ANSWER_PATTERN.search(annotation_block)
        if ans_match:
            answer_formula = ans_match.group(1).strip()

        # Detect topics for NoOp injection
        topics = cls._detect_topics(question_template)

        return TemplateSpec(
            template_id=template_id,
            question_template=question_template,
            answer_template=answer_annotated,
            variables=variables,
            conditions=conditions,
            answer_formula=answer_formula,
            original_question=data.get('question', ''),
            original_answer=data.get('answer', ''),
            original_id=data.get('id_orig', 0),
            detected_topics=topics
        )

    @classmethod
    def _parse_variables(cls, template: str) -> Dict[str, VariableSpec]:
        """Extract variable definitions from template placeholders."""
        variables = {}
        for match in cls.VARIABLE_PATTERN.finditer(template):
            content = match.group(1)
            parts = content.split(',')
            var_name = parts[0].strip()
            default_value = parts[1].strip() if len(parts) > 1 else None

            # Determine if numeric ($ prefix in init block or numeric default)
            is_numeric = False
            try:
                if default_value:
                    float(default_value)
                    is_numeric = True
            except (ValueError, TypeError):
                pass

            variables[var_name] = VariableSpec(
                name=var_name,
                var_type='literal',
                params=default_value,
                is_numeric=is_numeric
            )
        return variables

    @classmethod
    def _parse_init_block(cls, init_block: str, variables: Dict[str, VariableSpec]):
        """Parse the #init: block to get variable specifications."""
        for line in init_block.strip().split('\n'):
            line = line.strip()
            if not line or not line.startswith('-'):
                continue

            line = line[1:].strip()  # Remove leading dash

            # Parse assignment: var = expression
            if '=' not in line:
                continue

            left, right = line.split('=', 1)
            var_name = left.strip().lstrip('$')
            right = right.strip()

            is_numeric = left.strip().startswith('$')

            # Check for range()
            range_match = cls.RANGE_PATTERN.match(right)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                step = int(range_match.group(3)) if range_match.group(3) else 1
                if var_name in variables:
                    variables[var_name].var_type = 'range'
                    variables[var_name].params = (start, end, step)
                    variables[var_name].is_numeric = True
                continue

            # Check for sample()
            sample_match = cls.SAMPLE_PATTERN.match(right)
            if sample_match:
                collection = sample_match.group(1).strip()
                if var_name in variables:
                    variables[var_name].var_type = 'sample'
                    variables[var_name].params = collection
                    variables[var_name].is_numeric = is_numeric
                continue

            # Direct list assignment
            if right.startswith('['):
                try:
                    # Safely evaluate the list
                    items = eval(right, {"__builtins__": {}}, {})
                    if var_name in variables:
                        variables[var_name].var_type = 'sample'
                        variables[var_name].params = items
                        variables[var_name].is_numeric = is_numeric
                except Exception:
                    # If eval fails, store as literal
                    if var_name in variables:
                        variables[var_name].var_type = 'literal'
                        variables[var_name].params = right
                continue

    @classmethod
    def _parse_conditions(cls, cond_block: str) -> List[str]:
        """Parse condition expressions."""
        conditions = []
        for line in cond_block.strip().split('\n'):
            line = line.strip()
            if line.startswith('-'):
                conditions.append(line[1:].strip())
        return conditions

    @classmethod
    def _detect_topics(cls, template: str) -> Set[str]:
        """Detect topic domains for NoOp distractor selection."""
        topics = set()
        template_lower = template.lower()

        topic_keywords = {
            'agriculture': ['eat', 'bake', 'fruit', 'cook', 'slice', 'apple', 'orange',
                           'vegetable', 'farm', 'harvest', 'plant', 'grow', 'food'],
            'commerce': ['buy', 'sell', 'cost', 'price', 'dollar', 'pay', 'money',
                        'store', 'shop', 'discount', 'spend', 'earn', 'profit'],
            'inventory': ['book', 'toy', 'card', 'stamp', 'collect', 'item', 'box',
                         'container', 'stack', 'pile', 'shelf'],
            'travel': ['drive', 'mile', 'speed', 'hour', 'trip', 'travel', 'distance',
                      'car', 'train', 'plane', 'walk', 'run', 'bike'],
            'time': ['day', 'week', 'month', 'hour', 'minute', 'year', 'schedule',
                    'calendar', 'morning', 'evening', 'night'],
            'nature': ['fog', 'rain', 'weather', 'ocean', 'river', 'lake', 'mountain',
                      'forest', 'animal', 'fish', 'bird'],
            'education': ['student', 'teacher', 'class', 'school', 'grade', 'exam',
                         'homework', 'study', 'learn', 'read', 'write'],
            'construction': ['build', 'house', 'room', 'wall', 'floor', 'area',
                            'perimeter', 'length', 'width', 'height'],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in template_lower for kw in keywords):
                topics.add(topic)

        return topics if topics else {'general'}


class ConditionEvaluator:
    """Evaluates template conditions."""

    @staticmethod
    def is_int(x) -> bool:
        """Check if value is effectively an integer."""
        try:
            return float(x) == int(x)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def divides(a, b) -> bool:
        """Check if a divides b evenly."""
        try:
            return b % a == 0
        except (ZeroDivisionError, TypeError):
            return False

    @classmethod
    def evaluate(cls, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a condition string with given variables."""
        # Create safe evaluation context
        safe_context = {
            'is_int': cls.is_int,
            'divides': cls.divides,
            'int': int,
            'float': float,
            'abs': abs,
            'min': min,
            'max': max,
            'True': True,
            'False': False,
        }
        safe_context.update(variables)

        try:
            return bool(eval(condition, {"__builtins__": {}}, safe_context))
        except Exception:
            return False


class TemplateEngine:
    """
    Main engine for generating problem instances from GSM-Symbolic templates.

    Handles variable resolution, condition checking, and answer computation.
    """

    def __init__(self, templates_dir: Optional[Path] = None, seed: Optional[int] = None):
        """
        Initialize the TemplateEngine.

        Args:
            templates_dir: Directory containing template JSON files
            seed: Random seed for reproducibility
        """
        self.templates_dir = templates_dir
        self.rng = random.Random(seed)
        self.name_provider = NameProvider(seed)
        self.templates: Dict[str, TemplateSpec] = {}
        self._current_seed = seed

    def load_templates(self, variant: str = 'symbolic') -> int:
        """
        Load all templates for a given variant.

        Args:
            variant: 'symbolic' (base), 'p1', or 'p2'

        Returns:
            Number of templates loaded
        """
        if self.templates_dir is None:
            raise ValueError("templates_dir not set")

        variant_dir = self.templates_dir / variant
        if not variant_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {variant_dir}")

        # Clear existing templates to prevent accumulation across variants
        self.templates.clear()

        count = 0
        for template_file in sorted(variant_dir.glob('*.json')):
            try:
                spec = TemplateParser.parse_template_file(template_file)
                spec.template_id = f"{variant}_{spec.template_id}"
                self.templates[spec.template_id] = spec
                count += 1
            except Exception as e:
                print(f"Warning: Failed to parse {template_file}: {e}")

        return count

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._current_seed = seed
        self.rng = random.Random(seed)
        self.name_provider.set_seed(seed)

    def generate_instance(self, template_id: str, instance_num: int,
                          max_attempts: int = 500) -> Optional[GeneratedProblem]:
        """
        Generate a single problem instance from a template.

        Args:
            template_id: ID of the template to use
            instance_num: Instance number for this generation
            max_attempts: Maximum attempts to find valid variable assignment

        Returns:
            GeneratedProblem or None if generation fails
        """
        spec = self.templates.get(template_id)
        if spec is None:
            raise ValueError(f"Template not found: {template_id}")

        for attempt in range(max_attempts):
            # Generate variable values
            variables = self._sample_variables(spec)

            # Check all conditions
            if not self._check_conditions(spec.conditions, variables):
                continue

            # Compute the answer
            answer_value = self._evaluate_answer(spec.answer_formula, variables)
            if answer_value is None:
                continue

            # Validate the answer (non-negative, reasonable range)
            if not self._validate_answer(answer_value):
                continue

            # Generate question and answer text
            question_text = self._render_template(spec.question_template, variables)
            answer_text = self._render_answer(spec.answer_template, variables, answer_value)

            return GeneratedProblem(
                id=f"{template_id}_{instance_num}",
                template_id=template_id,
                instance=instance_num,
                question=question_text,
                answer=answer_text,
                final_answer=answer_value,
                variables=variables.copy(),
                is_noop=False,
                original_question=spec.original_question,
                original_answer=spec.original_answer
            )

        return None

    def generate_batch(self, template_id: str, num_instances: int = 50,
                       start_seed: Optional[int] = None) -> List[GeneratedProblem]:
        """
        Generate multiple instances from a single template.

        Args:
            template_id: Template to generate from
            num_instances: Number of instances to generate
            start_seed: Starting seed for reproducibility

        Returns:
            List of GeneratedProblem objects
        """
        problems = []
        base_seed = start_seed if start_seed is not None else self._current_seed or 0

        for i in range(num_instances):
            # Use deterministic seed for each instance
            instance_seed = base_seed + i
            self.set_seed(instance_seed)

            problem = self.generate_instance(template_id, i)
            if problem:
                problems.append(problem)

        return problems

    def _sample_variables(self, spec: TemplateSpec) -> Dict[str, Any]:
        """Sample values for all template variables."""
        variables = {}

        for var_name, var_spec in spec.variables.items():
            if var_spec.var_type == 'range':
                start, end, step = var_spec.params
                values = list(range(start, end + 1, step))
                variables[var_name] = self.rng.choice(values) if values else start

            elif var_spec.var_type == 'sample':
                collection = var_spec.params

                # Handle built-in collections
                if isinstance(collection, str):
                    # Check for built-in collection names
                    if collection in BUILTIN_COLLECTIONS:
                        coll_data = BUILTIN_COLLECTIONS[collection]
                        if isinstance(coll_data, dict):
                            # Fraction collection - sample key, store value
                            key = self.rng.choice(list(coll_data.keys()))
                            variables[var_name] = coll_data[key]
                            # Also store the text form for rendering
                            variables[f"{var_name}_text"] = key
                        else:
                            variables[var_name] = self.rng.choice(coll_data)
                    elif collection == 'names' or collection.startswith('names_'):
                        # Use our NameProvider for any name collection
                        names = self.name_provider.get_names_for_problem(1)
                        variables[var_name] = names[0] if names else "Alex"
                    elif collection.startswith('['):
                        # String that looks like a list - try to parse it
                        try:
                            items = eval(collection, {"__builtins__": {}}, {})
                            if isinstance(items, list) and items:
                                variables[var_name] = self.rng.choice(items)
                            else:
                                variables[var_name] = collection
                        except Exception:
                            variables[var_name] = collection
                    else:
                        # Unknown collection, use the string as-is
                        variables[var_name] = collection
                elif isinstance(collection, list):
                    if collection:
                        variables[var_name] = self.rng.choice(collection)
                    else:
                        variables[var_name] = ""
                else:
                    variables[var_name] = collection

            else:  # literal
                if var_spec.params is not None:
                    try:
                        variables[var_name] = float(var_spec.params) if var_spec.is_numeric else var_spec.params
                    except ValueError:
                        variables[var_name] = var_spec.params
                elif var_name.startswith('name') or 'name' in var_name.lower():
                    # Default to name sampling for name-like variables
                    names = self.name_provider.get_names_for_problem(1)
                    variables[var_name] = names[0] if names else "Alex"

        return variables

    def _check_conditions(self, conditions: List[str], variables: Dict[str, Any]) -> bool:
        """Check if all conditions are satisfied."""
        for condition in conditions:
            if not ConditionEvaluator.evaluate(condition, variables):
                return False
        return True

    def _evaluate_answer(self, formula: str, variables: Dict[str, Any]) -> Optional[Any]:
        """Safely evaluate the answer formula."""
        if not formula:
            return None

        safe_context = {
            'int': int,
            'float': float,
            'abs': abs,
            'min': min,
            'max': max,
            'round': round,
            'divmod': divmod,
            'True': True,
            'False': False,
        }
        safe_context.update(variables)

        try:
            result = eval(formula, {"__builtins__": {}}, safe_context)
            # Convert to int if it's effectively an integer
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return result
        except Exception as e:
            return None

    def _validate_answer(self, answer: Any) -> bool:
        """Validate the computed answer meets constraints."""
        try:
            num_answer = float(answer)
            # Non-negative check (for physical quantities)
            if num_answer < 0:
                return False
            # Reasonable range check (avoid extreme values)
            if abs(num_answer) > 1_000_000:
                return False
            return True
        except (ValueError, TypeError):
            return True  # Non-numeric answers are allowed

    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Render a template string with variable values."""
        result = template

        # Replace {var,default} patterns
        def replace_var(match):
            content = match.group(1)
            parts = content.split(',')
            var_name = parts[0].strip()

            # Handle expression in variable name
            if var_name in variables:
                value = variables[var_name]
            else:
                # Try to evaluate as expression
                try:
                    safe_ctx = {'int': int, 'float': float, 'abs': abs}
                    safe_ctx.update(variables)
                    value = eval(var_name, {"__builtins__": {}}, safe_ctx)
                except:
                    # Fall back to default if available
                    value = parts[1].strip() if len(parts) > 1 else var_name

            # Format numbers nicely
            if isinstance(value, float):
                if value == int(value):
                    return str(int(value))
                return str(value)
            return str(value)

        result = re.sub(r'\{([^}]+)\}', replace_var, result)
        return result

    def _render_answer(self, template: str, variables: Dict[str, Any],
                       final_answer: Any) -> str:
        """Render the answer template with step-by-step solution."""
        if not template:
            return f"#### {final_answer}"

        # Render the template
        result = self._render_template(template, variables)

        # Handle GSM8K calculation markers <<expr=result>>
        def replace_calc(match):
            return match.group(0)  # Keep as-is for now

        # Ensure the final answer line is present
        if '####' not in result:
            result += f"\n#### {final_answer}"

        return result

    def get_template_topics(self, template_id: str) -> Set[str]:
        """Get detected topics for a template (used by NoiseInjector)."""
        spec = self.templates.get(template_id)
        return spec.detected_topics if spec else {'general'}

    def get_all_template_ids(self) -> List[str]:
        """Get all loaded template IDs."""
        return list(self.templates.keys())


if __name__ == "__main__":
    # Demo usage
    from pathlib import Path

    # Adjust path as needed
    templates_path = Path(__file__).parent.parent.parent / "external" / "ml-gsm-symbolic" / "templates"

    engine = TemplateEngine(templates_dir=templates_path, seed=42)

    # Load base templates
    count = engine.load_templates('symbolic')
    print(f"Loaded {count} symbolic templates")

    # Generate from first template
    template_ids = engine.get_all_template_ids()
    if template_ids:
        print(f"\nGenerating 3 instances from {template_ids[0]}:")
        problems = engine.generate_batch(template_ids[0], num_instances=3)
        for p in problems:
            print(f"\n--- Instance {p.instance} ---")
            print(f"Question: {p.question[:200]}...")
            print(f"Answer: {p.final_answer}")
            print(f"Topics: {engine.get_template_topics(p.template_id)}")
