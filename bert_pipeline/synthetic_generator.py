"""
Synthetic Question Generator for Rare Classes

Uses LLM templates to generate diverse questions for classes with
very few samples (1-20 samples).
"""

import random
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from collections import defaultdict


class SyntheticQuestionGenerator:
    """
    Generate synthetic questions for rare classes using templates and LLM assistance.

    For manual LLM generation, this class provides prompts and structure.
    For automated generation, it can use template-based generation.
    """

    # Question templates by coarse class
    TEMPLATES = {
        "ABBR": [
            "What does {term} stand for?",
            "What is the abbreviation for {phrase}?",
            "What is the full form of {abbr}?",
            "What does {abbr} mean?",
            "{abbr} stands for what?",
        ],
        "DESC": {
            "def": [
                "What is the definition of {concept}?",
                "Define {concept}.",
                "What does {concept} mean?",
                "Explain what {concept} is.",
                "What is meant by {concept}?",
            ],
            "desc": [
                "Describe {entity}.",
                "What can you tell me about {entity}?",
                "Give a description of {entity}.",
                "What are the characteristics of {entity}?",
            ],
            "manner": [
                "How does {action} work?",
                "In what way does {process} occur?",
                "How is {action} performed?",
                "What is the method for {process}?",
            ],
            "reason": [
                "Why did {event} happen?",
                "What caused {outcome}?",
                "What is the reason for {phenomenon}?",
                "Why does {situation} exist?",
            ],
        },
        "ENTY": {
            "animal": [
                "What type of animal is {creature}?",
                "What kind of animal is {creature}?",
                "What species is {creature}?",
            ],
            "body": [
                "What part of the body is {body_part}?",
                "Where is the {body_part} located?",
            ],
            "color": [
                "What color is {object}?",
                "What is the color of {entity}?",
            ],
            "cremat": [
                "What {creative_work} was created by {creator}?",
                "What did {creator} create?",
            ],
            "currency": [
                "What is the currency of {country}?",
                "What money do they use in {nation}?",
                "What is the monetary unit of {place}?",
            ],
            "dismed": [
                "What disease is {symptom} a symptom of?",
                "What condition causes {effect}?",
                "What illness has {characteristic}?",
            ],
            "event": [
                "What event happened in {year}?",
                "What occurred during {period}?",
                "What took place at {location}?",
            ],
            "food": [
                "What kind of food is {dish}?",
                "What type of cuisine is {food_item}?",
            ],
            "instru": [
                "What instrument does {musician} play?",
                "What musical instrument is used in {context}?",
            ],
            "lang": [
                "What language do people speak in {country}?",
                "What is the official language of {nation}?",
            ],
            "letter": [
                "What letter comes {position} in the alphabet?",
                "What is the {ordinal} letter of the alphabet?",
            ],
            "plant": [
                "What type of plant is {plant_name}?",
                "What kind of plant is {vegetation}?",
            ],
            "product": [
                "What product does {company} make?",
                "What does {company} manufacture?",
                "What is produced by {company}?",
            ],
            "religion": [
                "What religion has {characteristic}?",
                "What faith believes in {belief}?",
                "What are the followers of {figure} called?",
            ],
            "sport": [
                "What sport does {athlete} play?",
                "In what sport is {term} used?",
            ],
            "substance": [
                "What substance is {material}?",
                "What chemical is {compound}?",
            ],
            "symbol": [
                "What does {symbol} represent?",
                "What is {symbol} a symbol of?",
            ],
            "tech": [
                "What technology is used for {purpose}?",
                "What technical term means {definition}?",
            ],
            "termeq": [
                "What is another term for {word}?",
                "What is synonymous with {term}?",
                "What word means the same as {term}?",
            ],
            "veh": [
                "What vehicle does {person} drive?",
                "What type of vehicle is {transport}?",
            ],
            "word": [
                "What word starts with {letter}?",
                "What begins with {character}?",
            ],
            "other": [
                "What {category} is {entity}?",
                "What type of {thing} is {item}?",
            ],
        },
        "HUM": {
            "gr": [
                "What {group} is known for {characteristic}?",
                "What {organization} does {activity}?",
            ],
            "ind": [
                "Who is {person}?",
                "What person is known for {achievement}?",
                "Who {action}?",
            ],
            "desc": [
                "What kind of person is {description}?",
                "Describe {individual}.",
            ],
            "title": [
                "What is the title of {work}'s {role}?",
                "What position does {person} hold?",
            ],
        },
        "LOC": {
            "city": [
                "What city is in {region}?",
                "What is the city of {characteristic}?",
            ],
            "country": [
                "What country is {place} in?",
                "What nation has {feature}?",
            ],
            "mount": [
                "What is the highest mountain in {region}?",
                "What mountain range is in {area}?",
            ],
            "other": [
                "What is the location of {entity}?",
                "Where can you find {place}?",
                "In what place is {thing} located?",
            ],
            "state": [
                "What state is {city} in?",
                "What US state is known for {feature}?",
            ],
        },
        "NUM": {
            "count": [
                "How many {items} are there?",
                "What is the number of {things}?",
                "Count the {objects}.",
            ],
            "date": [
                "When did {event} occur?",
                "What year was {thing} {action}?",
                "On what date did {happening} happen?",
            ],
            "dist": [
                "How far is {place} from {origin}?",
                "What is the distance to {destination}?",
            ],
            "money": [
                "How much does {item} cost?",
                "What is the price of {product}?",
                "How much money is {amount}?",
            ],
            "ord": [
                "What is the {ordinal} {category}?",
                "What number is {position}?",
            ],
            "period": [
                "How long did {event} last?",
                "What was the duration of {activity}?",
                "How much time passed during {period}?",
            ],
            "perc": [
                "What percentage of {group} {condition}?",
                "What percent of {things} are {state}?",
            ],
            "speed": [
                "How fast does {thing} go?",
                "What is the speed of {object}?",
                "At what rate does {process} occur?",
            ],
            "temp": [
                "What is the temperature of {object}?",
                "How hot is {place}?",
                "What's the temperature in {location}?",
            ],
            "vol": [
                "What is the volume of {object}?",
                "How much does {container} hold?",
                "What's the size of {space}?",
            ],
            "other": [
                "How much/many is {quantity}?",
                "What amount of {thing} is there?",
            ],
        },
    }

    def __init__(self):
        self.generated_cache: Dict[str, List[str]] = {}

    def get_fine_label_templates(self, fine_label: str) -> List[str]:
        """Get templates for a specific fine-grained label."""
        parts = fine_label.split(":")
        if len(parts) != 2:
            return []

        coarse, fine = parts

        if coarse in self.TEMPLATES:
            coarse_templates = self.TEMPLATES[coarse]
            if isinstance(coarse_templates, list):
                return coarse_templates
            elif isinstance(coarse_templates, dict) and fine in coarse_templates:
                return coarse_templates[fine]

        return []

    def generate_from_template(
        self,
        fine_label: str,
        count: int = 50,
        placeholders: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """
        Generate questions using templates with placeholder substitution.

        Args:
            fine_label: The fine-grained label (e.g., "ENTY:currency")
            count: Number of questions to generate
            placeholders: Optional dict of placeholder names to value lists

        Returns:
            List of generated questions
        """
        templates = self.get_fine_label_templates(fine_label)
        if not templates:
            return []

        generated = []
        while len(generated) < count and templates:
            template = random.choice(templates)

            # If placeholders provided, substitute them
            if placeholders:
                try:
                    question = template.format(**{
                        k: random.choice(v) if isinstance(v, list) else v
                        for k, v in placeholders.items()
                    })
                    generated.append(question)
                except (KeyError, ValueError):
                    pass
            else:
                # Add template without substitution (user will fill)
                generated.append(template)

            if len(generated) >= count:
                break

        return generated[:count]

    def create_llm_prompt(
        self,
        fine_label: str,
        existing_questions: List[str],
        target_count: int = 50
    ) -> str:
        """
        Create a prompt for LLM-based question generation.

        Args:
            fine_label: The fine-grained label
            existing_questions: Current example questions
            target_count: How many new questions to generate

        Returns:
            Prompt string for LLM
        """
        parts = fine_label.split(":")
        if len(parts) != 2:
            return ""

        coarse, fine = parts
        coarse_desc = self._get_coarse_description(coarse)
        fine_desc = self._get_fine_description(fine_label)

        prompt = f"""I am training a classifier on the TREC question classification dataset.
The class {fine_label} has the following {len(existing_questions)} training example(s):

"""

        for i, q in enumerate(existing_questions, 1):
            prompt += f"{i}. {q}\n"

        prompt += f"""
Class Description:
- Coarse category: {coarse} ({coarse_desc})
- Fine-grained: {fine} ({fine_desc})

Please generate {target_count} completely new, diverse, open-domain questions that belong to this exact category.

Requirements:
1. Questions should be natural and casual, like a person would naturally ask
2. Avoid overly formal or academic language
3. Vary the question structure (start with different question words: What, Which, Who, Where, How, etc.)
4. Use diverse vocabulary - don't repeat the same words
5. Questions should be about different specific examples within this category
6. Each question must be unique and different from the examples above

Output format: Print each question on a new line, nothing else."""

        return prompt

    def _get_coarse_description(self, coarse: str) -> str:
        """Get human-readable description of coarse class."""
        descriptions = {
            "ABBR": "Abbreviation",
            "DESC": "Description/Definition",
            "ENTY": "Entity/Thing",
            "HUM": "Human/Person",
            "LOC": "Location",
            "NUM": "Numeric/Quantity",
        }
        return descriptions.get(coarse, coarse)

    def _get_fine_description(self, fine_label: str) -> str:
        """Get human-readable description of fine-grained label."""
        parts = fine_label.split(":")
        if len(parts) != 2:
            return fine_label

        coarse, fine = parts

        descriptions = {
            "ABBR:exp": "abbreviation/expansion",
            "ABBR:abb": "abbreviation/abbreviation",

            "DESC:def": "definition",
            "DESC:desc": "description",
            "DESC:manner": "manner/method",
            "DESC:reason": "reason",

            "ENTY:animal": "animal",
            "ENTY:body": "body part",
            "ENTY:color": "color",
            "ENTY:cremat": "creative work",
            "ENTY:currency": "currency/money",
            "ENTY:dismed": "disease/medicine",
            "ENTY:event": "event",
            "ENTY:food": "food",
            "ENTY:instru": "instrument",
            "ENTY:lang": "language",
            "ENTY:letter": "alphabet letter",
            "ENTY:plant": "plant",
            "ENTY:product": "product",
            "ENTY:religion": "religion",
            "ENTY:sport": "sport",
            "ENTY:substance": "substance/material",
            "ENTY:symbol": "symbol",
            "ENTY:tech": "technology",
            "ENTY:termeq": "term equivalent",
            "ENTY:veh": "vehicle",
            "ENTY:word": "word",
            "ENTY:other": "other entity",

            "HUM:gr": "group/organization",
            "HUM:ind": "individual",
            "HUM:desc": "description",
            "HUM:title": "title/position",

            "LOC:city": "city",
            "LOC:country": "country",
            "LOC:mount": "mountain",
            "LOC:other": "other location",
            "LOC:state": "state",

            "NUM:count": "count/number",
            "NUM:date": "date/time",
            "NUM:dist": "distance",
            "NUM:money": "money",
            "NUM:ord": "ordinal/rank",
            "NUM:period": "period/duration",
            "NUM:perc": "percentage",
            "NUM:speed": "speed",
            "NUM:temp": "temperature",
            "NUM:vol": "volume",
            "NUM:other": "other numeric",
        }

        return descriptions.get(fine_label, fine)

    def parse_llm_output(self, output: str) -> List[str]:
        """
        Parse LLM output into a list of questions.

        Args:
            output: Raw LLM output text

        Returns:
            List of questions
        """
        questions = []
        for line in output.strip().split("\n"):
            line = line.strip()
            # Remove numbering if present
            line = line.lstrip("0123456789.)- ")
            if line and len(line) > 5:  # Filter out very short lines
                questions.append(line)

        return questions

    def save_prompts_for_class(
        self,
        fine_label: str,
        existing_questions: List[str],
        target_count: int,
        output_dir: Path
    ):
        """
        Save prompt and instructions for manual LLM generation.

        Args:
            fine_label: The fine-grained label
            existing_questions: Current example questions
            target_count: How many new questions to generate
            output_dir: Directory to save prompt file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create prompt
        prompt = self.create_llm_prompt(fine_label, existing_questions, target_count)

        # Save prompt
        prompt_file = output_dir / f"prompt_{fine_label.replace(':', '_')}.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)

        print(f"Prompt saved to: {prompt_file}")

        # Create instruction file
        instruction_file = output_dir / f"instructions_{fine_label.replace(':', '_')}.txt"
        with open(instruction_file, 'w') as f:
            f.write(f"""GENERATION INSTRUCTIONS FOR {fine_label}

1. Copy the prompt from: {prompt_file.name}
2. Paste it into Claude or your preferred LLM
3. Review the generated questions
4. Save the accepted questions to: generated_{fine_label.replace(':', '_')}.txt
5. Format: One question per line

Expected output: {target_count} questions
Current examples: {len(existing_questions)}
""")

        print(f"Instructions saved to: {instruction_file}")

    def load_generated_questions(
        self,
        fine_label: str,
        input_dir: Path
    ) -> List[str]:
        """
        Load manually generated questions from file.

        Args:
            fine_label: The fine-grained label
            input_dir: Directory containing generated files

        Returns:
            List of generated questions
        """
        input_file = input_dir / f"generated_{fine_label.replace(':', '_')}.txt"

        if not input_file.exists():
            return []

        with open(input_file, 'r') as f:
            # Skip comment lines (starting with #) and empty lines
            questions = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

        return questions


if __name__ == "__main__":
    # Test the generator
    from .utils import load_trec_file, get_data_path, extract_coarse_label

    generator = SyntheticQuestionGenerator()

    # Test with a rare class
    data_path = get_data_path() / "train_5500.label"
    data = load_trec_file(data_path)

    # Find ENTY:currency samples (should be just 1)
    currency_samples = [text for label, text in data if label == "ENTY:currency"]

    print(f"Found {len(currency_samples)} ENTY:currency samples:")
    for sample in currency_samples:
        print(f"  - {sample}")

    # Create prompt
    prompt = generator.create_llm_prompt("ENTY:currency", currency_samples, 50)

    print("\n" + "=" * 60)
    print("LLM PROMPT:")
    print("=" * 60)
    print(prompt)
