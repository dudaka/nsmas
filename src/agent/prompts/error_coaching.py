"""
Error-specific coaching templates for the Reflector node.

Maps each ErrorType to targeted guidance that helps the LLM
understand and fix the specific issue.
"""

from typing import Dict

# Error coaching templates for all 7 error types from ErrorType enum
ERROR_COACHING: Dict[str, str] = {
    "SYNTAX": """## Syntax Error Guidance
Common causes:
1. Missing period (.) at end of fact or rule
2. Mismatched parentheses or brackets
3. Using Uppercase for constants (should be lowercase)
4. Using lowercase for variables (should be Uppercase)
5. Invalid predicate names (must start with lowercase letter)
6. Missing comma between body literals

Check the line number in the error message and look for these issues.
Remember: facts look like `pred(a, b).` and rules look like `head(X) :- body(X).`""",

    "GROUNDING": """## Grounding/Safety Error Guidance
The error mentions "unsafe variable" - a variable in the rule HEAD is not defined in the BODY.

Example of UNSAFE (wrong):
```asp
total(T) :- quantity(john, apples, A).  % T is never bound!
```

Example of SAFE (correct):
```asp
total(T) :- quantity(john, apples, A), T = @add(A, 0).  % T is bound via @add
```

RULES:
1. Every variable in the head MUST appear in at least one positive literal in the body
2. Variables in @calc results ARE properly bound: `T = @add(A, B)` binds T
3. If a variable appears only in a comparison, it's unsafe
4. Check for typos in variable names (X vs Y)""",

    "TIMEOUT": """## Timeout Error Guidance
The solver exceeded the time limit. Common causes:

1. Using ranges like `value(1..1000000)` - NEVER do this!
2. Recursive rules without proper termination
3. Choice rules creating exponential model space
4. Not using @calc hooks for arithmetic

SOLUTION:
- All arithmetic MUST use @calc hooks: `T = @add(A, B)` not `T = A + B`
- Never enumerate large number ranges
- Remove any choice rules `{...}` unless absolutely necessary
- Simplify the logic - use intermediate predicates""",

    "UNSAT": """## UNSAT (Unsatisfiable) Error Guidance
No valid model exists - the constraints are contradictory.

Common causes:
1. Same entity/attribute assigned different values
2. Impossible constraint (e.g., X > 10 and X < 5)
3. Circular negative dependencies
4. Hard constraint `:-` that can never be satisfied

DEBUG STEPS:
1. Check if you're deriving the same predicate with different values
2. Look for hard constraints (rules starting with `:-`) that might conflict
3. Verify your arithmetic - maybe the numbers don't work out
4. Try removing constraints one at a time to find the conflict""",

    "NO_ANSWER": """## No Answer Error Guidance
The program is satisfiable but no `final_answer(N)` or `solution(N)` was derived.

Common causes:
1. Missing `final_answer(V) :- ...` rule entirely
2. The derivation chain doesn't reach `final_answer`
3. Typo in predicate name breaks the chain
4. Intermediate predicate has no matching facts

SOLUTION:
1. Ensure you have exactly ONE rule defining `final_answer(V) :- ...`
2. Trace your derivation: fact -> intermediate -> final_answer
3. Check spelling of ALL predicate names for typos
4. Verify each intermediate predicate actually produces values""",

    "AMBIGUOUS": """## Ambiguous Answer Error Guidance
Multiple different values were derived for `final_answer`.

Common causes:
1. Multiple rules deriving `final_answer` with different logic
2. Using choice rules `{...}` that create multiple models
3. Non-deterministic conditions allowing multiple paths
4. Missing constraints that should eliminate alternatives

SOLUTION:
1. Ensure only ONE derivation path to `final_answer`
2. Check for duplicate rules with same head but different bodies
3. Add constraints to eliminate unwanted models
4. Remove any choice rules unless you need them""",

    "RUNTIME": """## Runtime Error Guidance
A Python @calc hook failed during execution.

Common causes:
1. Division by zero in @div or @mod
2. Passing non-numeric values to arithmetic hooks
3. Overflow with very large numbers
4. Calling undefined functions

SOLUTION:
1. Check for division operations - ensure divisor cannot be 0
2. Verify all inputs to @calc hooks are integers
3. Add guards: only call @div when divisor is known non-zero
4. Use intermediate variables to debug complex calculations""",
}


def get_error_coaching(
    error_type: str,
    error_message: str,
    asp_code: str,
) -> str:
    """
    Get error-specific coaching for LLM reflection.

    Args:
        error_type: The ErrorType name (e.g., "SYNTAX", "GROUNDING")
        error_message: The raw error message from Clingo
        asp_code: The ASP code that caused the error

    Returns:
        Coaching text tailored to the specific error
    """
    # Get base coaching for error type
    coaching = ERROR_COACHING.get(error_type, ERROR_COACHING.get("RUNTIME", ""))

    # Add context-specific advice based on error message
    context_hints = []

    if "line" in error_message.lower():
        context_hints.append(
            "The error mentions a specific line number - review that line carefully."
        )

    if "undefined" in error_message.lower():
        context_hints.append(
            "A predicate or atom is undefined - check spelling and ensure it's declared."
        )

    if "unsafe" in error_message.lower():
        # Extract variable name if present
        import re
        match = re.search(r"'(\w+)'", error_message)
        if match:
            var_name = match.group(1)
            context_hints.append(
                f"The variable '{var_name}' is unsafe - ensure it appears in a positive body literal."
            )

    if "division" in error_message.lower() or "zero" in error_message.lower():
        context_hints.append(
            "Division by zero detected - add a guard to prevent dividing by zero."
        )

    # Combine coaching with context hints
    if context_hints:
        context_section = "\n\n## Specific Issues Detected\n" + "\n".join(
            f"- {hint}" for hint in context_hints
        )
        coaching += context_section

    return coaching
