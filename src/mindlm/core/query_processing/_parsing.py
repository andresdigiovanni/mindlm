import re


def parse_numbered_list(text: str) -> list[str]:
    lines = [
        re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip() for line in text.splitlines()
    ]
    return [line for line in lines if line]
