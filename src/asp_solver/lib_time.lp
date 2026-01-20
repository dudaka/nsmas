% =============================================================================
% lib_time.lp - Simplified State-Transition Library for NS-MAS
% =============================================================================
%
% This library provides a simplified temporal model for arithmetic word problems.
% It uses step indexing rather than full Event Calculus to avoid grounding
% explosion while maintaining the ability to reason about sequential operations.
%
% Design Principles:
%   1. No global inertia axiom - state derived through operations
%   2. Step indexing (0, 1, 2, ...) for sequential operations
%   3. T=0 as default for static problems (no temporal reasoning needed)
%   4. Localized frame axioms only when explicitly needed
%
% =============================================================================

% =============================================================================
% Step Definition and Ordering
% =============================================================================

% Define the valid step range for a problem
% The LLM should declare: max_step(N) where N is the number of operations
step(0).
step(T+1) :- step(T), max_step(Max), T < Max.

% Step ordering
before(T1, T2) :- step(T1), step(T2), T1 < T2.
after(T1, T2) :- before(T2, T1).
immediately_before(T, T+1) :- step(T), step(T+1).

% First and last steps
first_step(0) :- step(0).
last_step(T) :- step(T), max_step(T).

% For static problems, use default step 0
% If no max_step is defined, assume static (single-state) problem
max_step(0) :- not has_multiple_steps.
has_multiple_steps :- max_step(N), N > 0.

% =============================================================================
% State Representation
% =============================================================================

% Primary state predicate: at(Step, Entity, Attribute, Value)
% Example: at(0, john, apples, 5) means "At step 0, John has 5 apples"

% For static problems, state at step 0 is the only state
% Example: at(john, apples, 5) is shorthand for at(0, john, apples, 5)
at(0, Entity, Attr, Val) :- at(Entity, Attr, Val), not has_multiple_steps.

% =============================================================================
% Operations and State Transitions
% =============================================================================

% Operation types that cause state transitions
% op(Step, OpType, Entity, Attribute, Amount)

% Addition operation: increase value at next step
at(T+1, Entity, Attr, NewVal) :-
    op(T, add_to, Entity, Attr, Amount),
    at(T, Entity, Attr, OldVal),
    NewVal = @add(OldVal, Amount).

% Subtraction operation: decrease value at next step
at(T+1, Entity, Attr, NewVal) :-
    op(T, sub_from, Entity, Attr, Amount),
    at(T, Entity, Attr, OldVal),
    NewVal = @sub(OldVal, Amount).

% Multiplication operation: multiply value at next step
at(T+1, Entity, Attr, NewVal) :-
    op(T, mul_by, Entity, Attr, Factor),
    at(T, Entity, Attr, OldVal),
    NewVal = @mul(OldVal, Factor).

% Division operation: divide value at next step
at(T+1, Entity, Attr, NewVal) :-
    op(T, div_by, Entity, Attr, Divisor),
    at(T, Entity, Attr, OldVal),
    NewVal = @div(OldVal, Divisor).

% Set operation: directly set value at next step (overwrite)
at(T+1, Entity, Attr, NewVal) :-
    op(T, set_to, Entity, Attr, NewVal).

% =============================================================================
% Transfer Operations (L2 Semantic Actions)
% =============================================================================

% Transfer from one entity to another
% transfer(Step, From, To, Attribute, Amount)
% Results in: From loses Amount, To gains Amount

at(T+1, From, Attr, NewVal) :-
    transfer(T, From, To, Attr, Amount),
    at(T, From, Attr, OldVal),
    NewVal = @sub(OldVal, Amount).

at(T+1, To, Attr, NewVal) :-
    transfer(T, From, To, Attr, Amount),
    at(T, To, Attr, OldVal),
    NewVal = @add(OldVal, Amount).

% =============================================================================
% Localized Frame Axiom (Persistence)
% =============================================================================

% State persists to next step ONLY if no operation affects it
% This is opt-in: the LLM must declare which entities need persistence

% Mark entities/attributes that are modified at each step
modified(T, Entity, Attr) :- op(T, _, Entity, Attr, _).
modified(T, Entity, Attr) :- transfer(T, Entity, _, Attr, _).
modified(T, Entity, Attr) :- transfer(T, _, Entity, Attr, _).

% Persistence rule: carry forward unmodified state
% Only applies if the entity is marked as persistent
at(T+1, Entity, Attr, Val) :-
    at(T, Entity, Attr, Val),
    step(T+1),
    persistent(Entity, Attr),
    not modified(T, Entity, Attr).

% By default, entities involved in operations are persistent
persistent(Entity, Attr) :- at(0, Entity, Attr, _), has_multiple_steps.

% =============================================================================
% Final State Extraction
% =============================================================================

% Get the final value of an entity's attribute
final_value(Entity, Attr, Val) :-
    at(T, Entity, Attr, Val),
    last_step(T).

% For static problems (no temporal steps)
final_value(Entity, Attr, Val) :-
    at(Entity, Attr, Val),
    not has_multiple_steps.

% =============================================================================
% Constraints and Validation
% =============================================================================

% Uniqueness: At each step, each entity-attribute pair has at most one value
:- at(T, Entity, Attr, V1), at(T, Entity, Attr, V2), V1 != V2.

% Step validity: Operations only at valid steps
:- op(T, _, _, _, _), not step(T).
:- transfer(T, _, _, _, _), not step(T).

% Causality: Operations must have prior state to act on
:- op(T, OpType, Entity, Attr, _),
   OpType != set_to,
   not at(T, Entity, Attr, _).

:- transfer(T, From, To, Attr, _), not at(T, From, Attr, _).
:- transfer(T, From, To, Attr, _), not at(T, To, Attr, _).

% =============================================================================
% Soft Constraints (Error Predicates for Feedback)
% =============================================================================

% Negative value warning for physical quantities
error(negative_value, Entity, Attr, T) :-
    at(T, Entity, Attr, Val),
    physical_quantity(Attr),
    Val < 0.

% Insufficient funds/quantity for transfer
error(insufficient_amount, From, Attr, T) :-
    transfer(T, From, To, Attr, Amount),
    at(T, From, Attr, CurrentVal),
    CurrentVal < Amount.

% Mark common physical quantities (can be extended)
physical_quantity(count).
physical_quantity(quantity).
physical_quantity(amount).
physical_quantity(items).
physical_quantity(apples).
physical_quantity(oranges).
physical_quantity(dollars).
physical_quantity(cents).
physical_quantity(people).
physical_quantity(distance).
physical_quantity(weight).
