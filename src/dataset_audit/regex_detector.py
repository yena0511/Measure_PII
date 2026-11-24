"""Count PII matches with the Subramani et al. (2023) regular expressions (or the v2 config that replaces Email and Phone)."""
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RegexPattern:
    name: str
    label: str
    regex: re.Pattern
    must_contain: str | None = None
    candidate_start: re.Pattern | None = None
    replacements: tuple = ()  # ((compiled pattern, replacement), ...) applied to the text before matching
    exclude_line: re.Pattern | None = None  # drop a match when its line matches this
    value_group: str | None = None  # yield this named group when it took part in the match

    def find(self, text):
        """Yield the matched strings, in order."""
        for old, new in self.replacements:
            text = old.sub(new, text)
        for match in self._matches(text):
            if self.exclude_line is not None:
                start = text.rfind("\n", 0, match.start()) + 1
                end = text.find("\n", match.end())
                if self.exclude_line.search(text, start, len(text) if end < 0 else end):
                    continue
            value = match.group(self.value_group) if self.value_group else None
            yield value if value is not None else match.group(0)

    def _matches(self, text):
        if self.must_contain and self.must_contain not in text:
            return
        if self.candidate_start is None:
            yield from self.regex.finditer(text)
            return

        # Same matches as finditer, without running the pattern at every position.
        pos = 0
        while candidate := self.candidate_start.search(text, pos):
            match = self.regex.match(text, candidate.start())
            if match:
                yield match
                pos = match.end()
            else:
                pos = candidate.start() + 1

    def count(self, text):
        return sum(1 for _ in self.find(text))


class RegexDetector:
    def __init__(self, patterns_file):
        config = yaml.safe_load(Path(patterns_file).read_text(encoding="utf-8"))
        self.source = config["source"]
        self.modified_patterns = config.get("modified_patterns", {})
        self.patterns = [
            RegexPattern(
                name=name,
                label=spec["label"],
                regex=re.compile(spec["regex"]),
                must_contain=spec.get("must_contain"),
                candidate_start=re.compile(spec["candidate_start"]) if "candidate_start" in spec else None,
                replacements=tuple((re.compile(r["pattern"]), r["replace"]) for r in spec.get("replacements", [])),
                exclude_line=re.compile(spec["exclude_line"]) if "exclude_line" in spec else None,
                value_group=spec.get("value_group"),
            )
            for name, spec in config["patterns"].items()
        ]

    @property
    def labels(self):
        return [p.label for p in self.patterns]

    def find(self, text):
        """Return {label: [matched strings]} for each pattern that matched."""
        found = {}
        for pattern in self.patterns:
            values = list(pattern.find(text))
            if values:
                found[pattern.label] = values
        return found

    def count(self, text):
        """Return {label: match count} for each pattern that matched."""
        return {label: len(values) for label, values in self.find(text).items()}
