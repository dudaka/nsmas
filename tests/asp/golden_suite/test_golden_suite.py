"""
Golden Suite - 20 Manually Translated GSM-Symbolic Base Problems.

Each test case represents a manually verified translation from natural
language math word problem to ASP program, serving as:
1. Regression tests for the ASP solver
2. Example translations for LLM training
3. Coverage verification for the ontology

Problem Types:
- Simple arithmetic (add, sub, mul, div)
- Percentages and ratios
- Unit conversion
- Multi-step calculations
- Rate × time problems
- Geometry (area/perimeter)
- Comparison via multipliers
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.asp_solver.solver import ASPSolver


@pytest.fixture
def solver():
    """Create a solver instance for testing."""
    return ASPSolver(timeout_ms=10000)


# =============================================================================
# Golden Suite Test Cases
# =============================================================================

GOLDEN_SUITE = [
    # -------------------------------------------------------------------------
    # 1. Simple Addition with Multiplier (GSM 0020)
    # -------------------------------------------------------------------------
    {
        "id": "gs01_accumulation",
        "question": """John picks 4 bananas on Wednesday. Then he picks 6 bananas
        on Thursday. On Friday, he picks triple the number of bananas he did
        on Wednesday. How many bananas does John have?""",
        "answer": 22,  # 4 + 6 + (3 * 4) = 22
        "asp": """
            % Facts: quantities picked each day
            picked(wednesday, 4).
            picked(thursday, 6).
            multiplier(friday, 3).  % Triple of Wednesday

            % Friday's amount is 3x Wednesday
            picked(friday, R) :- picked(wednesday, W), multiplier(friday, M), R = @mul(W, M).

            % Sum all days
            wed_thu(S) :- picked(wednesday, W), picked(thursday, T), S = @add(W, T).
            total(T) :- wed_thu(WT), picked(friday, F), T = @add(WT, F).

            final_answer(T) :- total(T).
        """,
    },
    # -------------------------------------------------------------------------
    # 2. Simple Subtraction - Purchase and Change (GSM 0030)
    # -------------------------------------------------------------------------
    {
        "id": "gs02_purchase",
        "question": """David has $12.48 and wants to buy 16 bolts. Each bolt costs
        $0.03 (3 cents). How much money does David have left after paying?""",
        "answer": 12,  # 1248 - (16 * 3) = 1248 - 48 = 1200 cents = $12
        "asp": """
            % Using cents to avoid decimals
            initial_cents(1248).  % $12.48 in cents
            quantity(16).
            price_cents(3).  % $0.03 per bolt

            % Calculate total cost
            total_cost(C) :- quantity(Q), price_cents(P), C = @mul(Q, P).

            % Calculate remaining
            remaining_cents(R) :- initial_cents(I), total_cost(C), R = @sub(I, C).

            % Convert back to dollars (integer division by 100)
            remaining_dollars(D) :- remaining_cents(R), D = @div(R, 100).

            final_answer(D) :- remaining_dollars(D).
        """,
    },
    # -------------------------------------------------------------------------
    # 3. Multiplication - Yearly Earnings (GSM 0070)
    # -------------------------------------------------------------------------
    {
        "id": "gs03_yearly_earnings",
        "question": """A builder works for 4 weeks every month and for 6 days
        every week. If he gets paid $50 every day, how much does he earn
        if he works for a year?""",
        "answer": 14400,  # 4 * 6 * 50 * 12 = 14400
        "asp": """
            weeks_per_month(4).
            days_per_week(6).
            pay_per_day(50).
            months_per_year(12).

            % Days per month
            days_per_month(D) :- weeks_per_month(W), days_per_week(Dw), D = @mul(W, Dw).

            % Monthly earnings
            monthly_earnings(E) :- days_per_month(D), pay_per_day(P), E = @mul(D, P).

            % Yearly earnings
            yearly_earnings(Y) :- monthly_earnings(M), months_per_year(N), Y = @mul(M, N).

            final_answer(Y) :- yearly_earnings(Y).
        """,
    },
    # -------------------------------------------------------------------------
    # 4. Division - Flower Bed Planning (GSM 0045)
    # -------------------------------------------------------------------------
    {
        "id": "gs04_division",
        "question": """Pat has a flower bed that is 111 feet long. She needs to
        leave 3 feet between every plant. Pat already owns 17 flowers. Each
        plant costs $6. How much will Pat spend to fill her flower bed?""",
        "answer": 120,  # (111/3 - 17) * 6 = (37 - 17) * 6 = 20 * 6 = 120
        "asp": """
            bed_length(111).
            spacing(3).  % feet between plants (simplified from 1.5)
            owned(17).
            cost_per_plant(6).

            % Plants needed = length / spacing
            plants_needed(N) :- bed_length(L), spacing(S), N = @div(L, S).

            % Plants to buy = needed - owned
            plants_to_buy(B) :- plants_needed(N), owned(O), B = @sub(N, O).

            % Total cost
            total_cost(C) :- plants_to_buy(B), cost_per_plant(P), C = @mul(B, P).

            final_answer(C) :- total_cost(C).
        """,
    },
    # -------------------------------------------------------------------------
    # 5. Percentage - Combined Groups (GSM 0015)
    # -------------------------------------------------------------------------
    {
        "id": "gs05_percentage",
        "question": """Jennifer's dog has 8 puppies, 3 of which have spots.
        Brandon's dog has 12 puppies, 4 of which have spots. What percentage
        of all the puppies have spots?""",
        "answer": 35,  # (3+4) / (8+12) * 100 = 7/20 * 100 = 35%
        "asp": """
            puppies(jennifer, 8).
            spotted(jennifer, 3).
            puppies(brandon, 12).
            spotted(brandon, 4).

            % Total puppies
            total_puppies(T) :- puppies(jennifer, J), puppies(brandon, B), T = @add(J, B).

            % Total spotted
            total_spotted(S) :- spotted(jennifer, Sj), spotted(brandon, Sb), S = @add(Sj, Sb).

            % Percentage (spotted * 100 / total)
            percentage(P) :- total_spotted(S), total_puppies(T),
                             S100 = @mul(S, 100), P = @div(S100, T).

            final_answer(P) :- percentage(P).
        """,
    },
    # -------------------------------------------------------------------------
    # 6. Ratio - Coffee Recipe (GSM 0005)
    # -------------------------------------------------------------------------
    {
        "id": "gs06_ratio",
        "question": """Katy makes coffee using teaspoons of sugar and cups of
        water in the ratio of 7:13. If she used a total of 120 teaspoons of
        sugar and cups of water, calculate the number of teaspoons of sugar.""",
        "answer": 42,  # 7/(7+13) * 120 = 7/20 * 120 = 42
        "asp": """
            ratio_sugar(7).
            ratio_water(13).
            total_parts(120).

            % Total ratio
            total_ratio(R) :- ratio_sugar(S), ratio_water(W), R = @add(S, W).

            % Sugar = (sugar_ratio * total) / total_ratio
            sugar_amount(A) :- ratio_sugar(S), total_parts(T), total_ratio(R),
                               ST = @mul(S, T), A = @div(ST, R).

            final_answer(A) :- sugar_amount(A).
        """,
    },
    # -------------------------------------------------------------------------
    # 7. Rate × Time with Bonus (GSM 0060)
    # -------------------------------------------------------------------------
    {
        "id": "gs07_rate_time",
        "question": """Watson works a 10-hour shift each day, five days a week.
        He earns $10 per hour and gets a $300 bonus each week if the company
        performs well. How much did Watson make in April (4 weeks) if the
        company performed well for the whole month?""",
        "answer": 3200,  # (10 * 10 * 5 * 4) + (300 * 4) = 2000 + 1200 = 3200
        "asp": """
            hours_per_day(10).
            days_per_week(5).
            hourly_rate(10).
            weekly_bonus(300).
            weeks_in_month(4).

            % Daily earnings
            daily_earnings(D) :- hours_per_day(H), hourly_rate(R), D = @mul(H, R).

            % Weekly base earnings
            weekly_base(W) :- daily_earnings(D), days_per_week(Days), W = @mul(D, Days).

            % Monthly base earnings
            monthly_base(M) :- weekly_base(W), weeks_in_month(N), M = @mul(W, N).

            % Total bonuses
            total_bonus(B) :- weekly_bonus(Wb), weeks_in_month(N), B = @mul(Wb, N).

            % Total earnings
            total_earnings(T) :- monthly_base(M), total_bonus(B), T = @add(M, B).

            final_answer(T) :- total_earnings(T).
        """,
    },
    # -------------------------------------------------------------------------
    # 8. Unit Conversion + Percentage (GSM 0000)
    # -------------------------------------------------------------------------
    {
        "id": "gs08_unit_conversion",
        "question": """Benny saw a 10-foot shark with 2 6-inch remoras attached.
        What percentage of the shark's body length is the combined length
        of the remoras?""",
        "answer": 10,  # 2*6 = 12 inches = 1 foot; 1/10 * 100 = 10%
        "asp": """
            shark_feet(10).
            remora_count(2).
            remora_inches(6).
            inches_per_foot(12).

            % Total remora length in inches
            total_remora_inches(I) :- remora_count(N), remora_inches(R), I = @mul(N, R).

            % Convert to feet
            total_remora_feet(F) :- total_remora_inches(I), inches_per_foot(Ipf), F = @div(I, Ipf).

            % Shark length in feet (already in feet)
            % Calculate percentage
            percentage(P) :- total_remora_feet(Rf), shark_feet(Sf),
                             Rf100 = @mul(Rf, 100), P = @div(Rf100, Sf).

            final_answer(P) :- percentage(P).
        """,
    },
    # -------------------------------------------------------------------------
    # 9. Multi-step with Fraction and Doubling (GSM 0055)
    # -------------------------------------------------------------------------
    {
        "id": "gs09_multistep",
        "question": """Elise writes the alphabet (26 letters) in full twice,
        writes half of it once, then re-writes everything she has already
        written. How many letters has Elise written in total?""",
        "answer": 130,  # (26*2 + 26/2) * 2 = (52 + 13) * 2 = 65 * 2 = 130
        "asp": """
            alphabet_letters(26).
            full_writes(2).
            half_fraction(2).  % denominator for "half"

            % Letters from full writes
            full_letters(L) :- alphabet_letters(A), full_writes(N), L = @mul(A, N).

            % Letters from half write
            half_letters(H) :- alphabet_letters(A), half_fraction(F), H = @div(A, F).

            % Total before doubling
            before_double(T) :- full_letters(F), half_letters(H), T = @add(F, H).

            % After re-writing (doubling)
            after_double(D) :- before_double(T), D = @mul(T, 2).

            final_answer(D) :- after_double(D).
        """,
    },
    # -------------------------------------------------------------------------
    # 10. Comparison via Multiplier (GSM 0090)
    # -------------------------------------------------------------------------
    {
        "id": "gs10_comparison",
        "question": """Charlie has three times as many Facebook friends as Dorothy.
        James has four times as many friends as Dorothy. If Charlie has 12
        friends, how many friends does James have?""",
        "answer": 16,  # Dorothy = 12/3 = 4; James = 4 * 4 = 16
        "asp": """
            charlie_multiplier(3).  % Charlie = 3 * Dorothy
            james_multiplier(4).    % James = 4 * Dorothy
            charlie_friends(12).

            % Find Dorothy's friends
            dorothy_friends(D) :- charlie_friends(C), charlie_multiplier(M), D = @div(C, M).

            % Find James's friends
            james_friends(J) :- dorothy_friends(D), james_multiplier(M), J = @mul(D, M).

            final_answer(J) :- james_friends(J).
        """,
    },
    # -------------------------------------------------------------------------
    # 11. Opportunity Cost (GSM 0065)
    # -------------------------------------------------------------------------
    {
        "id": "gs11_opportunity_cost",
        "question": """Jackie is deciding whether to do her taxes herself or hire
        an accountant. If she does them herself, she'll lose 3 hours of
        freelance work at $35/hour. The accountant charges $90. How much more
        money will she have if she hires the accountant?""",
        "answer": 15,  # 3 * 35 - 90 = 105 - 90 = 15
        "asp": """
            hours_lost(3).
            hourly_rate(35).
            accountant_fee(90).

            % Lost income if DIY
            lost_income(L) :- hours_lost(H), hourly_rate(R), L = @mul(H, R).

            % Savings = lost_income - accountant_fee
            savings(S) :- lost_income(L), accountant_fee(F), S = @sub(L, F).

            final_answer(S) :- savings(S).
        """,
    },
    # -------------------------------------------------------------------------
    # 12. Inventory Calculation (GSM 0075)
    # -------------------------------------------------------------------------
    {
        "id": "gs12_inventory",
        "question": """There are 3 red balls ($9 each), 11 blue balls ($5 each),
        and 25 green balls ($3 each) in the store. How much will the store
        receive after all balls are sold?""",
        "answer": 157,  # 3*9 + 11*5 + 25*3 = 27 + 55 + 75 = 157
        "asp": """
            balls(red, 3, 9).
            balls(blue, 11, 5).
            balls(green, 25, 3).

            % Revenue per color
            revenue(Color, R) :- balls(Color, Qty, Price), R = @mul(Qty, Price).

            % Total revenue (sum all colors)
            red_blue(RB) :- revenue(red, R), revenue(blue, B), RB = @add(R, B).
            total_revenue(T) :- red_blue(RB), revenue(green, G), T = @add(RB, G).

            final_answer(T) :- total_revenue(T).
        """,
    },
    # -------------------------------------------------------------------------
    # 13. Splitting with Difference (GSM 0085)
    # -------------------------------------------------------------------------
    {
        "id": "gs13_splitting",
        "question": """Sam and Harry have 100 feet of fence. They agree to split
        it with Harry getting 60 feet more than Sam. How much does Sam get?""",
        "answer": 20,  # Let S = Sam. S + (S+60) = 100 => 2S = 40 => S = 20
        "asp": """
            total_fence(100).
            difference(60).  % Harry gets this much more

            % Sam's share = (total - difference) / 2
            sam_share(S) :- total_fence(T), difference(D),
                            TD = @sub(T, D), S = @div(TD, 2).

            final_answer(S) :- sam_share(S).
        """,
    },
    # -------------------------------------------------------------------------
    # 14. Tiered Pricing (GSM 0010)
    # -------------------------------------------------------------------------
    {
        "id": "gs14_tiered_pricing",
        "question": """To make a call from a payphone, you pay $0.25 per minute
        for the first 16 minutes, then $0.20 per minute after that. How much
        does a 36-minute call cost? (Answer in dollars)""",
        "answer": 8,  # 16*0.25 + 20*0.20 = 4 + 4 = 8
        "asp": """
            % Using cents to avoid decimals
            tier1_rate(25).   % cents per minute
            tier1_limit(16).  % minutes
            tier2_rate(20).   % cents per minute
            total_minutes(36).

            % Tier 1 cost
            tier1_cost(C) :- tier1_rate(R), tier1_limit(L), C = @mul(R, L).

            % Tier 2 minutes
            tier2_minutes(M) :- total_minutes(T), tier1_limit(L), M = @sub(T, L).

            % Tier 2 cost
            tier2_cost(C) :- tier2_rate(R), tier2_minutes(M), C = @mul(R, M).

            % Total in cents
            total_cents(T) :- tier1_cost(C1), tier2_cost(C2), T = @add(C1, C2).

            % Convert to dollars
            total_dollars(D) :- total_cents(C), D = @div(C, 100).

            final_answer(D) :- total_dollars(D).
        """,
    },
    # -------------------------------------------------------------------------
    # 15. Area and Perimeter (GSM 0050)
    # -------------------------------------------------------------------------
    {
        "id": "gs15_geometry",
        "question": """The area of Billie's rectangular bedroom is 360 square feet.
        If the length is 3 yards (= 9 feet), what is the perimeter in feet?""",
        "answer": 98,  # Width = 360/9 = 40; Perimeter = 2*(9+40) = 2*49 = 98
        "asp": """
            area(360).
            length_yards(3).
            feet_per_yard(3).

            % Convert length to feet
            length_feet(L) :- length_yards(Ly), feet_per_yard(F), L = @mul(Ly, F).

            % Calculate width
            width(W) :- area(A), length_feet(L), W = @div(A, L).

            % Calculate perimeter = 2 * (length + width)
            sum_sides(S) :- length_feet(L), width(W), S = @add(L, W).
            perimeter(P) :- sum_sides(S), P = @mul(S, 2).

            final_answer(P) :- perimeter(P).
        """,
    },
    # -------------------------------------------------------------------------
    # 16. Fraction of Cost + Deficit (GSM 0035)
    # -------------------------------------------------------------------------
    {
        "id": "gs16_fraction_deficit",
        "question": """John is raising money for a $300 school trip. The school
        will cover half the cost. John has $50. How much more does he need?""",
        "answer": 100,  # School pays 150, John has 50, needs 300-150-50 = 100
        "asp": """
            trip_cost(300).
            school_fraction(2).  % 1/2 = divide by 2
            john_has(50).

            % School contribution
            school_pays(S) :- trip_cost(T), school_fraction(F), S = @div(T, F).

            % Total covered so far
            covered(C) :- school_pays(S), john_has(J), C = @add(S, J).

            % Amount still needed
            needed(N) :- trip_cost(T), covered(C), N = @sub(T, C).

            final_answer(N) :- needed(N).
        """,
    },
    # -------------------------------------------------------------------------
    # 17. Multi-category Sum (GSM 0095)
    # -------------------------------------------------------------------------
    {
        "id": "gs17_categories",
        "question": """There are 6 students playing tennis and twice that number
        playing volleyball. There are 16 boys and 22 girls playing soccer.
        How many students are there in total?""",
        "answer": 56,  # 6 + 12 + (16+22) = 6 + 12 + 38 = 56
        "asp": """
            tennis(6).
            volleyball_mult(2).  % twice tennis
            soccer_boys(16).
            soccer_girls(22).

            % Volleyball count
            volleyball(V) :- tennis(T), volleyball_mult(M), V = @mul(T, M).

            % Soccer total
            soccer(S) :- soccer_boys(B), soccer_girls(G), S = @add(B, G).

            % Tennis + volleyball
            tennis_volleyball(TV) :- tennis(T), volleyball(V), TV = @add(T, V).

            % Total students
            total_students(Tot) :- tennis_volleyball(TV), soccer(S), Tot = @add(TV, S).

            final_answer(Tot) :- total_students(Tot).
        """,
    },
    # -------------------------------------------------------------------------
    # 18. Nested Percentages (GSM 0025)
    # -------------------------------------------------------------------------
    {
        "id": "gs18_nested_percent",
        "question": """In a class of 30 students, 20% are football players. Of the
        remaining students, 25% are cheerleaders. How many students total
        (football + cheerleaders) are leaving early for a game?""",
        "answer": 12,  # Football: 30*0.20=6; Remaining: 24; Cheerleaders: 24*0.25=6; Total: 12
        "asp": """
            total_students(30).
            football_percent(20).
            cheerleader_percent(25).

            % Football players
            football(F) :- total_students(T), football_percent(P),
                           TP = @mul(T, P), F = @div(TP, 100).

            % Remaining students
            remaining(R) :- total_students(T), football(F), R = @sub(T, F).

            % Cheerleaders (% of remaining)
            cheerleaders(C) :- remaining(R), cheerleader_percent(P),
                               RP = @mul(R, P), C = @div(RP, 100).

            % Total leaving
            leaving(L) :- football(F), cheerleaders(C), L = @add(F, C).

            final_answer(L) :- leaving(L).
        """,
    },
    # -------------------------------------------------------------------------
    # 19. Sequential State Changes
    # -------------------------------------------------------------------------
    {
        "id": "gs19_state_changes",
        "question": """Tom has 15 marbles. He gives 5 to his friend, then buys
        8 more, then loses 3. How many marbles does Tom have?""",
        "answer": 15,  # 15 - 5 + 8 - 3 = 15
        "asp": """
            initial(15).
            gives(5).
            buys(8).
            loses(3).

            % After giving
            after_give(A) :- initial(I), gives(G), A = @sub(I, G).

            % After buying
            after_buy(A) :- after_give(Ag), buys(B), A = @add(Ag, B).

            % After losing
            after_lose(A) :- after_buy(Ab), loses(L), A = @sub(Ab, L).

            final_answer(A) :- after_lose(A).
        """,
    },
    # -------------------------------------------------------------------------
    # 20. Complex Multi-step with Modulo
    # -------------------------------------------------------------------------
    {
        "id": "gs20_complex",
        "question": """A bakery has 137 cookies. They want to pack them into boxes
        of 12. How many cookies will be left over after packing as many
        full boxes as possible?""",
        "answer": 5,  # 137 % 12 = 5
        "asp": """
            total_cookies(137).
            box_size(12).

            % Remainder = total mod box_size
            leftover(L) :- total_cookies(T), box_size(B), L = @mod(T, B).

            final_answer(L) :- leftover(L).
        """,
    },
]


class TestGoldenSuite:
    """Test all 20 Golden Suite problems."""

    @pytest.mark.parametrize(
        "problem",
        GOLDEN_SUITE,
        ids=[p["id"] for p in GOLDEN_SUITE],
    )
    def test_golden_problem(self, solver, problem):
        """Test a single Golden Suite problem."""
        result = solver.solve(problem["asp"])

        # Check for success
        assert result.success, (
            f"Problem {problem['id']} failed:\\n"
            f"Question: {problem['question'][:100]}...\\n"
            f"Error: {result.to_feedback()}"
        )

        # Check answer matches expected
        assert result.answer == problem["answer"], (
            f"Problem {problem['id']} wrong answer:\\n"
            f"Expected: {problem['answer']}\\n"
            f"Got: {result.answer}\\n"
            f"Question: {problem['question'][:100]}..."
        )


class TestGoldenSuiteCoverage:
    """Test that the Golden Suite covers all key operations."""

    def test_addition_coverage(self, solver):
        """Verify addition operations work."""
        problems = [p for p in GOLDEN_SUITE if "@add" in p["asp"]]
        assert len(problems) >= 5, "Golden Suite should have at least 5 addition problems"

    def test_subtraction_coverage(self, solver):
        """Verify subtraction operations work."""
        problems = [p for p in GOLDEN_SUITE if "@sub" in p["asp"]]
        assert len(problems) >= 5, "Golden Suite should have at least 5 subtraction problems"

    def test_multiplication_coverage(self, solver):
        """Verify multiplication operations work."""
        problems = [p for p in GOLDEN_SUITE if "@mul" in p["asp"]]
        assert len(problems) >= 10, "Golden Suite should have at least 10 multiplication problems"

    def test_division_coverage(self, solver):
        """Verify division operations work."""
        problems = [p for p in GOLDEN_SUITE if "@div" in p["asp"]]
        assert len(problems) >= 8, "Golden Suite should have at least 8 division problems"

    def test_modulo_coverage(self, solver):
        """Verify modulo operation is covered."""
        problems = [p for p in GOLDEN_SUITE if "@mod" in p["asp"]]
        assert len(problems) >= 1, "Golden Suite should have at least 1 modulo problem"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
