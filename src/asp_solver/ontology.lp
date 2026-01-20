% =============================================================================
% ontology.lp - Layered Hybrid Ontology for NS-MAS
% =============================================================================
%
% This file defines the predicate schema for arithmetic word problems.
% It follows a Layered Hybrid approach:
%
%   L1: Core Physics - Properties of numbers, units, existence
%   L2: Semantic Actions - Linguistic events mapped to logic
%   L3: Arithmetic Ops - (Defined in lib_math.lp)
%
% Design Principle: Fixed predicate schema + Open vocabulary for constants
% The LLM MUST use these predicates; it cannot invent new ones.
% However, entity names (john, apples, etc.) are free-form constants.
%
% =============================================================================

% =============================================================================
% LAYER 1: CORE PHYSICS
% =============================================================================

% -----------------------------------------------------------------------------
% Entity Declaration
% -----------------------------------------------------------------------------
% entity(ID) - Declares that ID exists as a named entity in the problem
% Examples: entity(john). entity(mary). entity(store).

% Type classification for entities
entity_type(ID, person) :- person(ID).
entity_type(ID, place) :- place(ID).
entity_type(ID, object) :- object(ID).
entity_type(ID, group) :- group(ID).

% All typed entities are entities
entity(ID) :- entity_type(ID, _).

% -----------------------------------------------------------------------------
% Quantity (Primary State Predicate)
% -----------------------------------------------------------------------------
% quantity(Entity, Item, Value) - Entity has Value of Item
% Example: quantity(john, apples, 5).
%
% quantity(Entity, Item, Value, Unit) - With explicit unit
% Example: quantity(car, distance, 100, miles).

% Quantities establish entities and items implicitly
entity(E) :- quantity(E, _, _).
entity(E) :- quantity(E, _, _, _).
item(I) :- quantity(_, I, _).
item(I) :- quantity(_, I, _, _).

% Default unit is "units" (dimensionless count)
quantity(E, I, V, units) :- quantity(E, I, V), not has_explicit_unit(E, I).
has_explicit_unit(E, I) :- quantity(E, I, _, U), U != units.

% -----------------------------------------------------------------------------
% Value (Derived/Computed Values)
% -----------------------------------------------------------------------------
% value(ID, Number) - Associates a computed value with an identifier
% Use for intermediate calculations and final answers
% Example: value(total_cost, 150).

% -----------------------------------------------------------------------------
% Attribute (Static Properties)
% -----------------------------------------------------------------------------
% attr(Entity, Property, Value) - Static property of an entity
% Example: attr(car, speed, 60). attr(box, capacity, 100).

% Attributes establish entities
entity(E) :- attr(E, _, _).

% -----------------------------------------------------------------------------
% Relation (Comparative/Proportional)
% -----------------------------------------------------------------------------
% rel(Type, Entity1, Entity2, Factor) - Relationship between entities
% Types: ratio, multiple, fraction, percent
%
% Examples:
%   rel(multiple, john_apples, mary_apples, 3)  -> John has 3x Mary's apples
%   rel(percent, part, whole, 25)              -> Part is 25% of whole
%   rel(ratio, boys, total, 2)                 -> Boys:Total = 2:N

% Relation types
relation_type(ratio).
relation_type(multiple).
relation_type(fraction).
relation_type(percent).
relation_type(equals).
relation_type(greater_than).
relation_type(less_than).

% Derive values from relations
value(E1, V1) :-
    rel(multiple, E1, E2, Factor),
    value(E2, V2),
    V1 = @mul(V2, Factor).

value(E1, V1) :-
    rel(percent, E1, E2, Pct),
    value(E2, V2),
    V1 = @percent_of(Pct, V2).

% =============================================================================
% LAYER 2: SEMANTIC ACTIONS
% =============================================================================

% -----------------------------------------------------------------------------
% Transfer (Give/Receive/Buy/Sell)
% -----------------------------------------------------------------------------
% transfer(From, To, Item, Amount) - Static (single-step) transfer
% transfer(Step, From, To, Item, Amount) - Temporal transfer (see lib_time.lp)
%
% Example: transfer(john, mary, apples, 3) -> John gives 3 apples to Mary

% Effect of static transfer on quantities
quantity(From, Item, NewVal) :-
    transfer(From, To, Item, Amount),
    initial_quantity(From, Item, OldVal),
    NewVal = @sub(OldVal, Amount),
    not has_multiple_steps.

quantity(To, Item, NewVal) :-
    transfer(From, To, Item, Amount),
    initial_quantity(To, Item, OldVal),
    NewVal = @add(OldVal, Amount),
    not has_multiple_steps.

% Mark initial quantities for transfer calculations
initial_quantity(E, I, V) :- given_quantity(E, I, V).

% -----------------------------------------------------------------------------
% Combine (Aggregation)
% -----------------------------------------------------------------------------
% combine(Target, Source1, Source2, Item) - Add quantities together
% combine(Target, group(Sources), Item) - Sum all sources in a group
%
% Example: combine(total, john, mary, apples) -> total = john's + mary's apples

value(Target, Sum) :-
    combine(Target, S1, S2, Item),
    quantity(S1, Item, V1),
    quantity(S2, Item, V2),
    Sum = @add(V1, V2).

% Group combination using ASP aggregation
value(Target, Sum) :-
    combine(Target, group(G), Item),
    Sum = #sum{ V, E : group_member(G, E), quantity(E, Item, V) }.

% -----------------------------------------------------------------------------
% Partition (Division/Distribution)
% -----------------------------------------------------------------------------
% partition(Source, Target, Item, Parts) - Divide equally into Parts
% partition(Source, Target, Item, ratio(A,B)) - Divide by ratio
%
% Example: partition(total, share, cookies, 4) -> share = total / 4

value(Target, Share) :-
    partition(Source, Target, Item, Parts),
    quantity(Source, Item, Total),
    Parts > 0,
    Share = @div(Total, Parts).

% Ratio-based partition
value(Target, Share) :-
    partition(Source, Target, Item, ratio(MyPart, TotalParts)),
    quantity(Source, Item, Total),
    TotalParts > 0,
    Share = @ratio_scale(Total, MyPart, TotalParts).

% -----------------------------------------------------------------------------
% Rate (Speed/Price/Frequency)
% -----------------------------------------------------------------------------
% rate(Entity, Attribute, Value, Unit) - Rate definition
% Example: rate(car, speed, 60, mph). rate(worker, wage, 15, per_hour).

% Rate calculations
% Distance = Speed × Time
value(distance(E), D) :-
    rate(E, speed, S, _),
    duration(E, T),
    D = @mul(S, T).

% Cost = Price × Quantity
value(cost(E, I), C) :-
    rate(I, price, P, _),
    quantity(E, I, Q),
    C = @mul(P, Q).

% Total = Rate × Duration
value(total(E, A), T) :-
    rate(E, A, R, _),
    duration(E, D),
    T = @mul(R, D).

% -----------------------------------------------------------------------------
% Consume (Use up resources)
% -----------------------------------------------------------------------------
% consume(Entity, Item, Amount) - Entity uses up Amount of Item
% Example: consume(john, fuel, 10)

quantity(Entity, Item, NewVal) :-
    consume(Entity, Item, Amount),
    initial_quantity(Entity, Item, OldVal),
    NewVal = @sub(OldVal, Amount),
    not has_multiple_steps.

% -----------------------------------------------------------------------------
% Produce (Create resources)
% -----------------------------------------------------------------------------
% produce(Entity, Item, Amount) - Entity creates Amount of Item
% Example: produce(factory, widgets, 100)

quantity(Entity, Item, NewVal) :-
    produce(Entity, Item, Amount),
    initial_quantity(Entity, Item, OldVal),
    NewVal = @add(OldVal, Amount),
    not has_multiple_steps.

% When entity had no initial quantity, production creates it
quantity(Entity, Item, Amount) :-
    produce(Entity, Item, Amount),
    not initial_quantity(Entity, Item, _),
    not has_multiple_steps.

% =============================================================================
% ANSWER EXTRACTION
% =============================================================================

% -----------------------------------------------------------------------------
% Target Pointer Pattern
% -----------------------------------------------------------------------------
% target(QueryID) - Marks what we're solving for
% The solution is extracted from value(QueryID, V)

solution(V) :- target(ID), value(ID, V).

% Alternative: Direct final_answer specification
solution(V) :- final_answer(V).

% Extract from quantity if target references an entity-item pair
solution(V) :- target(qty(E, I)), quantity(E, I, V).

% Extract from final_value (temporal problems)
solution(V) :- target(final(E, A)), final_value(E, A, V).

% =============================================================================
% CONSTRAINTS
% =============================================================================

% Must have exactly one solution (for well-formed problems)
% Commented out by default; enable for strict validation
% :- #count{ V : solution(V) } != 1.

% Physical quantities cannot be negative (soft constraint via error predicate)
% Hard constraint version (uncomment if needed):
% :- quantity(E, I, V), V < 0, physical_item(I).

% Error predicate for negative quantities
error(negative_quantity, E, I, V) :-
    quantity(E, I, V),
    physical_item(I),
    V < 0.

% Default: all items are physical unless marked abstract
physical_item(I) :- item(I), not abstract_item(I).

% Mark abstract items that can be negative (temperature, profit, etc.)
% abstract_item(temperature).
% abstract_item(profit_loss).

% =============================================================================
% UTILITY PREDICATES
% =============================================================================

% Count entities
entity_count(N) :- N = #count{ E : entity(E) }.

% Count items
item_count(N) :- N = #count{ I : item(I) }.

% Sum all quantities of an item across entities
total_quantity(Item, Total) :-
    item(Item),
    Total = #sum{ V, E : quantity(E, Item, V) }.

% Check if entity has any quantity of an item
has_item(E, I) :- quantity(E, I, V), V > 0.
