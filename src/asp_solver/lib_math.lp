% =============================================================================
% lib_math.lp - Mathematical Operations Library for NS-MAS
% =============================================================================
%
% This library provides ASP rules that interface with Python @calc hooks.
% The Python functions are defined in context.py and passed via Context
% object during grounding.
%
% Design Principles:
%   1. No arithmetic ranges (e.g., value(1..1000000)) - causes memory explosion
%   2. All computations via Python hooks - exact arithmetic
%   3. Purely functional - no side effects or state transitions
%   4. Decimals handled via fixed-point (multiply by 100)
%
% Available Python hooks (from context.py):
%   @add, @sub, @mul, @div, @mod
%   @gt, @lt, @gte, @lte, @eq, @neq
%   @abs_val, @neg, @max_val, @min_val, @pow_val
%   @percent_of, @as_percent, @ratio_scale
%   @convert, @convert_div
%   @decimal_add, @decimal_sub, @decimal_mul, @decimal_div
%   @to_decimal, @from_decimal_whole, @from_decimal_frac
%
% Usage Pattern:
%   1. Assert a request fact: calc_add(5, 3).
%   2. Library derives result: calc_result(add, 5, 3, 8).
%   3. Use in your rules: answer(R) :- calc_result(add, 5, 3, R).
%
%   For comparisons, use Python hooks directly in rule bodies:
%      valid(X) :- value(X), @gt(X, 0) = 1.
%
% =============================================================================

% =============================================================================
% Core Arithmetic Operations
% =============================================================================
% Usage: Assert calc_add(A, B) to get calc_result(add, A, B, Result)

calc_result(add, A, B, R) :- calc_add(A, B), R = @add(A, B).
calc_result(sub, A, B, R) :- calc_sub(A, B), R = @sub(A, B).
calc_result(mul, A, B, R) :- calc_mul(A, B), R = @mul(A, B).
calc_result(div, A, B, R) :- calc_div(A, B), R = @div(A, B).
calc_result(mod, A, B, R) :- calc_mod(A, B), R = @mod(A, B).

% =============================================================================
% Advanced Operations (Single Argument)
% =============================================================================
% Usage: Assert do_abs(A) to get result_abs(A, R)

result_abs(A, R) :- do_abs(A), R = @abs_val(A).
result_neg(A, R) :- do_neg(A), R = @neg(A).

% =============================================================================
% Advanced Operations (Two Arguments)
% =============================================================================
% Usage: Assert do_max(A, B) to get result_max(A, B, R)

result_max(A, B, R) :- do_max(A, B), R = @max_val(A, B).
result_min(A, B, R) :- do_min(A, B), R = @min_val(A, B).
result_pow(A, B, R) :- do_pow(A, B), R = @pow_val(A, B).

% =============================================================================
% Percentage Operations
% =============================================================================
% Usage: Assert do_percent_of(P, W) to get result_percent_of(P, W, R)
%        where R = P% of W

result_percent_of(P, W, R) :- do_percent_of(P, W), R = @percent_of(P, W).
result_as_percent(Part, Whole, R) :- do_as_percent(Part, Whole), R = @as_percent(Part, Whole).
result_ratio(V, Num, Denom, R) :- do_ratio(V, Num, Denom), R = @ratio_scale(V, Num, Denom).

% =============================================================================
% Unit Conversion via Request
% =============================================================================
% Usage: Assert do_convert(V, Factor) to get result_convert(V, Factor, R)

result_convert(V, Factor, R) :- do_convert(V, Factor), R = @convert(V, Factor).
result_convert_div(V, Factor, R) :- do_convert_div(V, Factor), R = @convert_div(V, Factor).

% =============================================================================
% Common Unit Conversion Factors
% =============================================================================
% Usage: unit_factor(from_unit, to_unit, Factor)

% Length
unit_factor(feet, inches, 12).
unit_factor(yards, feet, 3).
unit_factor(miles, feet, 5280).
unit_factor(meters, centimeters, 100).
unit_factor(kilometers, meters, 1000).

% Weight
unit_factor(pounds, ounces, 16).
unit_factor(kilograms, grams, 1000).
unit_factor(tons, pounds, 2000).

% Time
unit_factor(hours, minutes, 60).
unit_factor(minutes, seconds, 60).
unit_factor(days, hours, 24).
unit_factor(weeks, days, 7).

% Money (for fixed-point: dollars -> cents)
unit_factor(dollars, cents, 100).

% =============================================================================
% Automatic Unit Conversion Rules
% =============================================================================
% Convert from one unit to another using defined factors
% Requires: quantity(Entity, Val, FromUnit) fact from user

converted_value(Entity, ToUnit, NewVal) :-
    quantity(Entity, Val, FromUnit),
    unit_factor(FromUnit, ToUnit, Factor),
    NewVal = @mul(Val, Factor).

converted_value(Entity, ToUnit, NewVal) :-
    quantity(Entity, Val, FromUnit),
    unit_factor(ToUnit, FromUnit, Factor),
    NewVal = @div(Val, Factor).
