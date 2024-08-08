# %%
from typing import Iterable, Any
import itertools
from collections.abc import Generator
import random

def random_make_parameter(
        param_list: dict[str, tuple[float, float]],
) -> Generator[dict[str, float], None, None]:
    """
    Generates a sequence of dictionaries, each containing randomly sampled values within specified ranges.

    Args:
        param_list (dict[str, tuple(float, float)]): A dictionary where each key is a parameter name, and the value is a tuple
                                                     representing the range (low, high) from which to sample the parameter's value.

    Yields:
        dict[str, float]: A dictionary with the same keys as `param_list`, where each value is a randomly sampled float
                          within the specified range for that key.

    Example:
        >>> param_list = {
        >>>     "param1": (0.0, 1.0),
        >>>     "param2": (10.0, 20.0),
        >>>     "param3": (100.0, 200.0),
        >>> }
        >>> generator = random_make_parameter_list(param_list)
        >>> next(generator)
        {'param1': 0.345, 'param2': 15.678, 'param3': 150.123}
        >>> next(generator)
        {'param1': 0.789, 'param2': 12.345, 'param3': 175.456}
    """
    while True:
        yield {key: random.uniform(low, high) for key, (low, high) in param_list.items()}


def combine_dicts(
    arr_list: dict[str, list],
) -> Generator[dict[str, float], None, None]:
    """
    This function takes a dictionary of iterables and returns a generator of dictionaries
    containing all possible combinations of elements from the input iterables.

    Parameters:
    arr_list (Dict[str, Iterable]): A dictionary where keys are strings and values are iterables.

    Returns:
    Generator[Dict[str, float]]: A generator of dictionaries containing all combinations of elements.
    """
    keys = arr_list.keys()
    values = arr_list.values()
    combinations = itertools.product(*values)
    for combination in combinations:
        yield dict(zip(keys, combination))


if __name__ == "__main__":
    print("example 1----------------------------------")
    arr_list = {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]}
    result = combine_dicts(arr_list)
    print(list(result))


    print("example 2----------------------------------")
    param_list = {
        "param1": (0.0, 1.0),
        "param2": (10.0, 20.0),
        "param3": (100.0, 200.0),
    }
    generator = random_make_parameter(param_list)
    for _ in range(5):
        print(next(generator))

#%%

# from typing import Iterable, Dict, List

# def combine_dicts(d: Dict[str, Iterable[object]]) -> None:
#     # 関数の実装
#     pass

# example_dict: Dict[str, List[float]] = {
#     "key1": [1.0, 2.0, 3.0],
#     "key2": [4.0, 5.0, 6.0],
# }

# combine_dicts(example_dict)
