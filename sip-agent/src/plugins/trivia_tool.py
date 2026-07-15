"""
Trivia Tool Plugin
==================
A voice trivia game played over the phone. The agent asks general-knowledge
questions, checks the caller's spoken answers with fuzzy matching, and keeps
a running score for the duration of the call.

A deployment can override the built-in question bank by dropping a
``trivia.json`` file (same shape as QUESTION_BANK) into the data directory;
if the file is missing or malformed the built-in bank is used.

Usage in conversation:
User: "Let's play trivia"
LLM: [TOOL:TRIVIA:action=ask]

User: "I think it's Paris"
LLM: [TOOL:TRIVIA:action=answer,answer=Paris]

User: "How am I doing?"
LLM: [TOOL:TRIVIA:action=score]
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

# Built-in general-knowledge question bank. Each entry lists every accepted
# answer (synonyms, digit and word forms) since callers answer by voice.
QUESTION_BANK: List[Dict[str, Any]] = [
    {"question": "What is the capital of France?", "answers": ["paris"]},
    {"question": "How many planets are in our solar system?", "answers": ["eight", "8"]},
    {"question": "What is the largest ocean on Earth?", "answers": ["pacific"]},
    {"question": "Who wrote Romeo and Juliet?", "answers": ["shakespeare", "william shakespeare"]},
    {"question": "Which planet is known as the red planet?", "answers": ["mars"]},
    {"question": "What element has the chemical symbol O?", "answers": ["oxygen"]},
    {"question": "How many continents are there on Earth?", "answers": ["seven", "7"]},
    {"question": "What is the largest mammal in the world?", "answers": ["blue whale", "whale"]},
    {"question": "Who painted the Mona Lisa?", "answers": ["leonardo da vinci", "da vinci", "leonardo"]},
    {"question": "What is the capital of Japan?", "answers": ["tokyo"]},
    {"question": "What is the fastest land animal?", "answers": ["cheetah"]},
    {"question": "What common substance has the chemical formula H two O?", "answers": ["water"]},
    {"question": "How many days are in a leap year?", "answers": ["366", "three hundred sixty six", "three hundred and sixty six"]},
    {"question": "Who was the first president of the United States?", "answers": ["george washington", "washington"]},
    {"question": "What is the tallest mountain on Earth?", "answers": ["everest", "mount everest"]},
    {"question": "How many sides does a hexagon have?", "answers": ["six", "6"]},
    {"question": "What is the longest river in the world?", "answers": ["nile", "amazon"]},
    {"question": "What do you call water in its solid frozen form?", "answers": ["ice"]},
    {"question": "What is the currency of Japan?", "answers": ["yen"]},
    {"question": "What is the smallest planet in our solar system?", "answers": ["mercury"]},
    {"question": "What gas do plants absorb from the air?", "answers": ["carbon dioxide", "co2", "c o 2"]},
    {"question": "Who wrote the Harry Potter books?", "answers": ["rowling", "jk rowling", "j k rowling", "joanne rowling"]},
    {"question": "What is the capital of Italy?", "answers": ["rome"]},
    {"question": "How many legs does a spider have?", "answers": ["eight", "8"]},
    {"question": "Which animal is known as the king of the jungle?", "answers": ["lion"]},
    {"question": "What color do you get when you mix blue and yellow?", "answers": ["green"]},
    {"question": "Which musical instrument has eighty eight keys?", "answers": ["piano"]},
    {"question": "What is the largest hot desert in the world?", "answers": ["sahara"]},
    {"question": "Which bird is famous for mimicking human speech?", "answers": ["parrot"]},
    {"question": "Which country is home to the Great Barrier Reef?", "answers": ["australia"]},
    {"question": "How many minutes are in an hour?", "answers": ["sixty", "60"]},
    {"question": "Which metal is liquid at room temperature?", "answers": ["mercury"]},
    {"question": "How often are the summer Olympic games held?", "answers": ["every four years", "four years", "4 years", "four", "4"]},
    {"question": "What do bees make from nectar?", "answers": ["honey"]},
    {"question": "What is the capital of England?", "answers": ["london"]},
    {"question": "What is seven plus six?", "answers": ["thirteen", "13"]},
    {"question": "Which animal is known as man's best friend?", "answers": ["dog"]},
    {"question": "Which planet in our solar system is famous for its rings?", "answers": ["saturn"]},
    {"question": "What language is primarily spoken in Brazil?", "answers": ["portuguese"]},
    {"question": "How many colors are in a rainbow?", "answers": ["seven", "7"]},
]

_STATE_KEY = "trivia"


def _normalize_answer(text: str) -> str:
    """Lowercase, keep only letters, digits and spaces, collapse whitespace."""
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_correct(caller_answer: str, accepted_answers: List[str]) -> bool:
    """Fuzzy match: accept when a normalized accepted answer is contained in
    the caller's normalized answer, or vice versa ("The Paris" matches "paris")."""
    caller = _normalize_answer(caller_answer)
    if not caller:
        return False
    for accepted in accepted_answers:
        normalized = _normalize_answer(accepted)
        if normalized and (normalized in caller or caller in normalized):
            return True
    return False


def _validate_bank(raw: Any) -> List[Dict[str, Any]]:
    """Return the entries of `raw` that have the expected shape, or []."""
    if not isinstance(raw, list):
        return []
    bank: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        question = entry.get("question")
        answers = entry.get("answers")
        if (isinstance(question, str) and question.strip()
                and isinstance(answers, list)
                and any(isinstance(a, str) and a.strip() for a in answers)):
            bank.append({
                "question": question.strip(),
                "answers": [a.strip() for a in answers if isinstance(a, str) and a.strip()],
            })
    return bank


def _load_bank(config: Optional[Any]) -> List[Dict[str, Any]]:
    """Load a question-bank override from data_dir/trivia.json when present;
    fail-open to the built-in bank on any problem."""
    if config is not None:
        try:
            path = Path(config.data_dir) / "trivia.json"
            if path.exists():
                bank = _validate_bank(json.loads(path.read_text(encoding="utf-8")))
                if bank:
                    logger.info(f"Trivia bank override loaded: {len(bank)} questions from {path}")
                    return bank
                logger.warning(f"Trivia override {path} has no valid questions - using built-in bank")
        except Exception as e:
            logger.warning(f"Failed to load trivia override: {e} - using built-in bank")
    return QUESTION_BANK


class TriviaTool(BaseTool):
    """Play a trivia game with the caller."""

    name = "TRIVIA"
    description = ("Play a trivia game with the caller: ask a question, "
                   "check their answer, or report the running score")
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode

    parameters = {
        "action": {
            "type": "string",
            "description": "What to do: 'ask' a new question, check an 'answer', or report the 'score'",
            "required": False,
            "default": "ask",
        },
        "answer": {
            "type": "string",
            "description": "The caller's answer, used with action=answer",
            "required": False,
            "default": "",
        },
    }

    def __init__(self, assistant):
        super().__init__(assistant)
        self._bank = _load_bank(self.config)

    def _get_state(self) -> Optional[Dict[str, Any]]:
        """Per-call game state; lives on the session, never on the singleton tool."""
        session = getattr(self.assistant, "session", None) if self.assistant else None
        if session is None:
            return None
        state = session.tool_state.get(_STATE_KEY)
        if state is None:
            state = {"current": None, "asked": [], "score": 0, "rounds": 0}
            session.tool_state[_STATE_KEY] = state
        return state

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            state = self._get_state()
            if state is None:
                return ToolResult(status=ToolStatus.FAILED,
                                  message="I can only play trivia during a call.")

            action = str(params.get("action") or "ask").strip().lower()
            if action == "answer":
                return self._check_answer(state, str(params.get("answer") or ""))
            if action == "score":
                return self._report_score(state)
            return self._ask_question(state)
        except Exception as e:
            logger.error(f"Trivia error: {e}", exc_info=True)
            return ToolResult(status=ToolStatus.FAILED,
                              message="Something went wrong with the trivia game.")

    def _ask_question(self, state: Dict[str, Any]) -> ToolResult:
        asked = state.get("asked") or []
        unasked = [i for i in range(len(self._bank)) if i not in asked]
        if not unasked:
            # Bank exhausted: start over so the game can keep going.
            asked = []
            unasked = list(range(len(self._bank)))

        idx = random.choice(unasked)
        asked.append(idx)
        state["asked"] = asked
        state["current"] = idx

        question = self._bank[idx]["question"]
        log_event(logger, logging.INFO, f"Trivia question asked: {question}",
                  event="trivia_ask")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=question,
            data={"question": question, "score": state["score"], "rounds": state["rounds"]},
        )

    def _check_answer(self, state: Dict[str, Any], answer: str) -> ToolResult:
        current = state.get("current")
        if current is None:
            return ToolResult(status=ToolStatus.FAILED,
                              message="Ask me for a question first.")
        if not answer.strip():
            # Don't burn the round on an empty transcription.
            return ToolResult(status=ToolStatus.FAILED,
                              message="I didn't catch your answer. Try again.")

        entry = self._bank[current]
        correct = _is_correct(answer, entry["answers"])
        state["rounds"] += 1
        if correct:
            state["score"] += 1
        state["current"] = None

        score, rounds = state["score"], state["rounds"]
        if correct:
            message = f"That is right! Your score is {score} out of {rounds}."
        else:
            message = (f"Not quite - the answer is {entry['answers'][0]}. "
                       f"Your score is {score} out of {rounds}.")

        log_event(logger, logging.INFO,
                  f"Trivia answer {'correct' if correct else 'wrong'}: {answer}",
                  event="trivia_answer")
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={"correct": correct, "score": score, "rounds": rounds},
        )

    def _report_score(self, state: Dict[str, Any]) -> ToolResult:
        score, rounds = state["score"], state["rounds"]
        if rounds == 0:
            message = "We haven't played any rounds yet. Ask me for a question to start."
        else:
            message = f"You have {score} right out of {rounds} questions so far."
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=message,
            data={"score": score, "rounds": rounds},
        )
