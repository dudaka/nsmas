"""
Few-shot examples for ASP generation.

These examples are drawn from the golden suite of manually verified
GSM-Symbolic problems.
"""

from typing import List

FEW_SHOT_EXAMPLES = [
    {
        "id": "accumulation",
        "question": "John picks 4 bananas on Wednesday. Then he picks 6 bananas on Thursday. On Friday, he picks triple the number of bananas he did on Wednesday. How many bananas does John have in total?",
        "asp": """% Facts: quantities picked each day
picked(wednesday, 4).
picked(thursday, 6).
multiplier(friday, 3).

% Friday's amount is 3x Wednesday
picked(friday, F) :- picked(wednesday, W), multiplier(friday, M), F = @mul(W, M).

% Sum Wednesday and Thursday
wed_thu(S) :- picked(wednesday, W), picked(thursday, T), S = @add(W, T).

% Total all three days
total(T) :- wed_thu(WT), picked(friday, F), T = @add(WT, F).

final_answer(T) :- total(T).""",
        "answer": 22,
    },
    {
        "id": "percentage",
        "question": "Jennifer's dog has 8 puppies, 3 of which have spots. Brandon's dog has 12 puppies, 4 of which have spots. What percentage of all the puppies have spots?",
        "asp": """% Facts about puppies
puppies(jennifer, 8).
spotted(jennifer, 3).
puppies(brandon, 12).
spotted(brandon, 4).

% Total puppies from both
total_puppies(T) :- puppies(jennifer, J), puppies(brandon, B), T = @add(J, B).

% Total spotted puppies
total_spotted(S) :- spotted(jennifer, Sj), spotted(brandon, Sb), S = @add(Sj, Sb).

% Percentage = (spotted * 100) / total
percentage(P) :- total_spotted(S), total_puppies(T), S100 = @mul(S, 100), P = @div(S100, T).

final_answer(P) :- percentage(P).""",
        "answer": 35,
    },
    {
        "id": "sequential_operations",
        "question": "Tom has 15 marbles. He gives 5 to his friend, then buys 8 more, then loses 3. How many marbles does Tom have now?",
        "asp": """% Initial state
initial(15).
gives(5).
buys(8).
loses(3).

% Step 1: After giving away
after_give(A) :- initial(I), gives(G), A = @sub(I, G).

% Step 2: After buying more
after_buy(A) :- after_give(Ag), buys(B), A = @add(Ag, B).

% Step 3: After losing some
after_lose(A) :- after_buy(Ab), loses(L), A = @sub(Ab, L).

final_answer(A) :- after_lose(A).""",
        "answer": 15,
    },
    {
        "id": "multiplication_with_rate",
        "question": "Sarah earns $12 per hour. She works 8 hours on Monday and 6 hours on Tuesday. How much does she earn in total?",
        "asp": """% Facts
hourly_rate(12).
hours(monday, 8).
hours(tuesday, 6).

% Total hours worked
total_hours(H) :- hours(monday, M), hours(tuesday, T), H = @add(M, T).

% Total earnings = rate * hours
earnings(E) :- hourly_rate(R), total_hours(H), E = @mul(R, H).

final_answer(E) :- earnings(E).""",
        "answer": 168,
    },
    {
        "id": "comparison_difference",
        "question": "A store has 45 red apples and 28 green apples. How many more red apples than green apples are there?",
        "asp": """% Facts
apples(red, 45).
apples(green, 28).

% Difference = red - green
difference(D) :- apples(red, R), apples(green, G), D = @sub(R, G).

final_answer(D) :- difference(D).""",
        "answer": 17,
    },
]


def get_few_shot_examples(num_examples: int = 3) -> str:
    """
    Get formatted few-shot examples for the system prompt.

    Args:
        num_examples: Number of examples to include (default 3)

    Returns:
        Formatted string with examples ready for prompt injection
    """
    examples = FEW_SHOT_EXAMPLES[:num_examples]

    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(
            f"""### Example {i}: {ex['id'].replace('_', ' ').title()}
**Question:** {ex['question']}

**ASP Code:**
```asp
{ex['asp'].strip()}
```

**Expected Answer:** {ex['answer']}"""
        )

    return "\n\n".join(formatted)


def get_example_by_id(example_id: str) -> dict:
    """
    Get a specific example by its ID.

    Args:
        example_id: The example identifier

    Returns:
        The example dict, or None if not found
    """
    for ex in FEW_SHOT_EXAMPLES:
        if ex["id"] == example_id:
            return ex
    return None
