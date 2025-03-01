"""
# math_eval.py
"""

import os
import re
import traceback
from typing import AsyncIterator, Dict, List, Optional

import sympy
from asynciolimiter import StrictLimiter
from loguru import logger
from openai import AsyncOpenAI
from sympy.parsing.latex import parse_latex
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential


class MathEvaluator:
    """
    either pass in an openai api key, or set the OPENAI_API_KEY environment variable.
    export OPENAI_API_KEY="sk-proj-..."

    usage:
        ```python
            evaluator = MathEvaluator()
            result = await evaluator.is_correct(correct_answer="4", proposed_answer="4")
            assert result
        ```
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        rate_limit: float = 10000 / 60,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the MathEvaluator with dataset paths, OpenAI client, and processing configurations.

        Args:
            train_data_path (str): Path to the training dataset.
            test_data_path (str): Path to the testing dataset.
            model_name (str): OpenAI model name to use for judging equality.
            rate_limit (float): Rate limit for asynchronous operations.
            api_key (Optional[str]): OpenAI API key. If None, it will use the OPENAI_API_KEY environment variable.
        """
        self.model_name = model_name
        self.rate_limiter = StrictLimiter(rate_limit)
        self.openai_client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    async def judge_equality(self, expr1: str, expr2: str) -> bool:
        """
        Determines if two mathematical expressions are equivalent using the OpenAI client.

        Args:
            expr1 (str): Generated answer.
            expr2 (str): True answer.

        Returns:
            bool: True if equivalent, False otherwise.
        """
        prompt = f"""
        As a mathematical judge, your job is to determine if a generated answer matches the true answer.

        Generated Answer: {expr1}
        True Answer: {expr2}

        Please consider all possible mathematical transformations and simplifications.
        Ignore all other details in the generated answer, and only consider the final answer in it.
        Respond with only 'Equivalent' or 'Not Equivalent'.
        """
        await self.rate_limiter.wait()
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                n=1,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            return result.lower() == "equivalent"
        except Exception as e:
            logger.error(f"error in judge_equality: {e}")
            traceback.print_exc()
            return False

    async def is_correct(self, correct_answer: str, proposed_answer: str) -> bool:
        """
        checks if the provided answer is correct.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=120),
            reraise=True,
        ):
            with attempt:
                extracted_answer = self.get_answer_expr(proposed_answer)

                if self.is_equiv(extracted_answer, correct_answer):
                    return True
                elif self.sympy_match(extracted_answer, correct_answer):
                    return True
                else:
                    return await self.judge_equality(extracted_answer, correct_answer)
                
    @staticmethod
    def is_correct_sync(correct_answer: str, proposed_answer: str) -> bool:
        """
        checks if the provided answer is correct.
        """
        extracted_answer = MathEvaluator.get_answer_expr(proposed_answer)

        if MathEvaluator.is_equiv(extracted_answer, correct_answer):
            return True
        elif MathEvaluator.sympy_match(extracted_answer, correct_answer):
            return True
        else:
            return False

    async def is_correct_anywhere(self, correct_answer: str, proposed_answer: str) -> bool:
        """
        checks if the correct answer appears anywhere in the proposed answer.
        """
        if await self.is_correct(correct_answer, proposed_answer):
            return True

        boxed_expressions = self.extract_boxed_expressions(proposed_answer)

        for expr in boxed_expressions:
            extracted_answer = self.remove_boxed(expr)
            if self.is_equiv(extracted_answer, correct_answer):
                return True
            elif self.sympy_match(extracted_answer, correct_answer):
                return True
            elif await self.judge_equality(extracted_answer, correct_answer):
                return True

        return False

    async def __call__(self, split: str) -> AsyncIterator[Dict]:
        """
        Allows the MathEvaluator to be called as an async generator.

        Args:
            split (str): The dataset split to use ('train' or 'test').

        Yields:
            Dict: The next item in the dataset.

        Raises:
            ValueError: If an invalid split is provided.
        """
        if split == "train":
            dataset = self.ds_train
        elif split == "test":
            dataset = self.ds_test
        else:
            raise ValueError("split must be 'train' or 'test'")

        for item in dataset:
            yield item

    @staticmethod
    def has_formatted_answer(answer: str) -> bool:
        """
        Checks if the answer contains a formatted solution.

        Args:
            answer (str): The answer string.

        Returns:
            bool: True if formatted answer exists, False otherwise.
        """
        try:
            if MathEvaluator.remove_boxed(MathEvaluator.last_boxed_only_string(answer)):
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def get_answer_expr(answer: str) -> str:
        """
        Extracts the mathematical expression from the answer.

        Args:
            answer (str): The answer string.

        Returns:
            str: Extracted expression.
        """
        try:
            answer = MathEvaluator.remove_boxed(MathEvaluator.last_boxed_only_string(answer))
        except Exception:
            answer = answer.split("\n")[-1]
        return answer

    @staticmethod
    def extract_boxed_expressions(string: str) -> List[str]:
        """
        extracts all \boxed{...} and \boxed ... expressions from the string.
        """
        boxed_expressions = []

        pattern_braces = r"\\boxed\s*\{([^}]*)\}"
        boxed_expressions += re.findall(pattern_braces, string)

        pattern_space = r"\\boxed\s+([^\s\$]+)"
        boxed_expressions += re.findall(pattern_space, string)

        return ["\\boxed{" + expr + "}" for expr in boxed_expressions]

    @staticmethod
    def remove_boxed(s: str) -> Optional[str]:
        """
        Removes the \boxed or \fbox formatting from a string.

        Args:
            s (str): The input string.

        Returns:
            Optional[str]: String without boxed formatting or None.
        """
        # pattern = r"\\boxed\s*{([^}]*)}"
        # return re.sub(pattern, r"\1", s, flags=re.DOTALL)

        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left) :]
        elif "\\boxed{" in s:
            left = "\\boxed{"
            assert s[: len(left)] == left
            assert s[-1] == "}"
            return s[len(left) : -1]
        elif "\\fbox{" in s:
            left = "\\fbox{"
            assert s[: len(left)] == left
            assert s[-1] == "}"
            return s[len(left) : -1]
        else:
            return s

    @staticmethod
    def last_boxed_only_string(string: str) -> Optional[str]:
        """
        Extracts the last boxed expression from a string.

        Args:
            string (str): The input string.

        Returns:
            Optional[str]: The last boxed expression or None.
        """
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None

    @staticmethod
    def is_equiv(str1: Optional[str], str2: Optional[str], verbose: bool = False) -> bool:
        """
        Checks if two strings are equivalent after normalization.

        Args:
            str1 (Optional[str]): First string.
            str2 (Optional[str]): Second string.
            verbose (bool): If True, prints the normalized strings.

        Returns:
            bool: True if equivalent, False otherwise.
        """
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = MathEvaluator.strip_string(str1)
            ss2 = MathEvaluator.strip_string(str2)
            if verbose:
                print(ss1, ss2)
            return ss1 == ss2
        except Exception:
            return str1 == str2

    @staticmethod
    def sympy_match(str1: str, str2: str) -> bool:
        """
        Checks if two mathematical expressions are equivalent using SymPy.

        Args:
            str1 (str): First expression.
            str2 (str): Second expression.

        Returns:
            bool: True if equivalent, False otherwise.
        """
        try:
            expr1 = parse_latex(str1)
            expr2 = parse_latex(str2)
            diff = sympy.simplify(expr1 - expr2)
            return diff == 0
        except Exception:
            return False

    @staticmethod
    def strip_string(string: str) -> str:
        """
        Normalizes a LaTeX string by removing unnecessary characters and formatting.

        Args:
            string (str): The input string.

        Returns:
            str: Normalized string.
        """
        string = string.replace("\n", "")
        string = string.replace("\\!", "")
        string = string.replace("\\\\", "\\")
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        string = string.replace("\\$", "")
        string = MathEvaluator.remove_right_units(string)
        string = string.replace("\\%", "")
        string = string.replace(r"\%", "")
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")

        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        string = MathEvaluator.fix_sqrt(string)
        string = string.replace(" ", "")
        string = MathEvaluator.fix_fracs(string)

        if string == "0.5":
            string = "\\frac{1}{2}"

        string = MathEvaluator.fix_a_slash_b(string)

        return string

    @staticmethod
    def fix_fracs(string: str) -> str:
        """
        Fixes improperly formatted fractions in a LaTeX string.

        Args:
            string (str): The input string.

        Returns:
            str: String with fixed fractions.
        """
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr.startswith("{"):
                    new_str += substr
                else:
                    if len(substr) < 2:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += f"{{{a}}}{{{b}}}{post_substr}"
                        else:
                            new_str += f"{{{a}}}{{{b}}}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += f"{{{a}}}{{{b}}}{post_substr}"
                        else:
                            new_str += f"{{{a}}}{{{b}}}"
        return new_str

    @staticmethod
    def fix_a_slash_b(string: str) -> str:
        """
        Converts a simple a/b format to LaTeX fraction if applicable.

        Args:
            string (str): The input string.

        Returns:
            str: Modified string with fractions fixed.
        """
        parts = string.split("/")
        if len(parts) != 2:
            return string
        a, b = parts
        try:
            a = int(a)
            b = int(b)
            if string == f"{a}/{b}":
                return f"\\frac{{{a}}}{{{b}}}"
            else:
                return string
        except ValueError:
            return string

    @staticmethod
    def remove_right_units(string: str) -> str:
        """
        Removes units described within \\text{ } at the end of the string.

        Args:
            string (str): The input string.

        Returns:
            str: String without units.
        """
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            if len(splits) == 2:
                return splits[0]
        return string

    @staticmethod
    def fix_sqrt(string: str) -> str:
        """
        Ensures that square roots in the string are properly formatted with braces.

        Args:
            string (str): The input string.

        Returns:
            str: String with fixed square roots.
        """
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if not split.startswith("{"):
                if len(split) < 1:
                    return string
                a = split[0]
                new_substr = f"\\sqrt{{{a}}}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string