from ml_retriever.verdict import (parse_magnitude, all_magnitudes, question_direction,
                                  yes_no_ground_truth, yes_no_predicted, comparison_ground_truth)

def test_parse_magnitude():
    assert parse_magnitude("$1,299") == 1299
    assert parse_magnitude("170 grams") == 170
    assert parse_magnitude("48MP main camera") == 48
    assert parse_magnitude("632 meters") == 632
    assert parse_magnitude("two to four days") is None

def test_all_magnitudes_in_order():
    assert all_magnitudes("$799, $799") == [799, 799]
    assert all_magnitudes("170 grams, 198 grams") == [170, 198]

def test_direction():
    assert question_direction("Does X have a higher price than Y?") == "greater"
    assert question_direction("Which laptop is lighter: A or B?") == "less"
    assert question_direction("Is X ahead of Y on price?") == "greater"

def test_yes_no_truth_equal_is_no():
    # both $799 -> "higher"? -> No (strict)
    assert yes_no_ground_truth("Does X have a higher price than Y?", "$799", "$799") == "no"

def test_yes_no_truth_and_pred():
    # iPhone 170g vs Pixel 198g, "higher weight"? -> No
    assert yes_no_ground_truth("Does X have a higher weight than Y?", "170 grams", "198 grams") == "no"
    assert yes_no_predicted("Does X have a higher weight than Y?", "170 grams, 198 grams") == "no"
    # a bare correct-value answer that does NOT resolve the verdict -> cannot predict
    assert yes_no_predicted("Does X have a higher price than Y?", "$799") is None

def test_comparison_winner():
    # lighter: 9.4 oz vs 10.6 oz -> first (Adidas) wins
    assert comparison_ground_truth("Which shoe is lighter: Adidas or Asics?",
                                   "Adidas", "9.4 ounces", "Asics", "10.6 ounces") == "Adidas"

def test_entity_model_number_not_parsed_as_value():
    # 'iPhone 16' / 'Pixel 9' model numbers must NOT be read as the weight value
    from ml_retriever.verdict import parse_magnitude
    assert parse_magnitude("The iPhone 16 weighs about 170 grams.", ["iPhone 16"]) == 170
    assert parse_magnitude("The Google Pixel 9 weighs about 198 grams.", ["Google Pixel 9"]) == 198
    # so the verdict is correct: 170 !> 198 -> "no"
    assert yes_no_ground_truth("Does X have a higher weight than Y?",
                               "The iPhone 16 weighs about 170 grams.",
                               "The Google Pixel 9 weighs about 198 grams.",
                               "iPhone 16", "Google Pixel 9") == "no"
