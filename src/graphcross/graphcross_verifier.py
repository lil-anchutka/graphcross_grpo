import json
import re
from typing import Dict
from base.data import Data
from base.verifier import Verifier


class GraphCrossVerifier(Verifier):
    """
    Verifier for GraphCross task.

    Assumptions:
    - data.answer is a dict
    - data.metadata is a dict

    IMPORTANT BEHAVIOR (as requested):
    - Formatting / extraction issues (can't find JSON, JSON parse error, etc.)
      RAISE ValueError immediately.
    - Semantic / constraint mismatches return False.
    """

    # extraction utilities

    def _strip_code_fences(self, s: str) -> str:
        s = re.sub(r"```(?:json|JSON)?", "", s)
        s = s.replace("```", "")
        return s.strip()

    def _extract_answer_region(self, text: str) -> str:
        m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text)
        return m.group(1).strip() if m else text.strip()

    def _find_first_json_object(self, text: str) -> str:
        start = text.find("{")
        if start == -1:
            return ""

        in_str = False
        esc = False
        depth = 0
        obj_start = -1

        for i in range(start, len(text)):
            ch = text[i]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and obj_start != -1:
                        return text[obj_start : i + 1]

        return ""

    def _cleanup_json_blob(self, blob: str) -> str:
        # Fix trailing commas like {"a":1,}
        return re.sub(r",\s*([}\]])", r"\1", blob).strip()

    def extract_answer(self, test_solution: str) -> Dict[str, str]:
        """
        Returns normalized dict[str,str].

        Raises ValueError on ANY formatting/extraction issue:
        - empty/non-string
        - no JSON found
        - JSON parse error
        - top-level not a dict
        """
        if not test_solution or not isinstance(test_solution, str):
            raise ValueError("Empty or non-string solution")

        region = self._extract_answer_region(test_solution)
        region = self._strip_code_fences(region)

        blob = self._find_first_json_object(region)
        if not blob:
            # fallback: sometimes it's pure JSON already
            blob = region.strip()
            if not (blob.startswith("{") and "}" in blob):
                raise ValueError("No JSON object found")

        blob = self._cleanup_json_blob(blob)

        try:
            obj = json.loads(blob)
        except Exception as e:
            raise ValueError(f"JSON parse error: {e!r}")

        if not isinstance(obj, dict):
            raise ValueError("Top-level JSON is not an object/dict")

        return {str(k): str(v) for k, v in obj.items()}

    def verify(self, data: Data, test_answer: str) -> bool:
        """
        Returns True/False for semantic correctness.

        NOTE: formatting issues propagate as ValueError from extract_answer()
        (so training can crash immediately if the model output is not parseable).
        """
        pred = self.extract_answer(test_answer)  # may raise ValueError

        gold = data.answer
        meta = data.metadata

        if not isinstance(gold, dict) or not isinstance(meta, dict):
            return False

        gold = {str(k): str(v) for k, v in gold.items()}
        pred = {str(k): str(v) for k, v in pred.items()}

        # Must match exactly the expected set of slots
        if set(pred.keys()) != set(gold.keys()):
            return False

        slots = meta.get("slots", {})
        candidates = meta.get("candidates", {})
        intersections = meta.get("intersections", [])

        if not isinstance(slots, dict) or not isinstance(candidates, dict) or not isinstance(intersections, list):
            return False

        try:
            slots = {str(k): int(v) for k, v in slots.items()}
            candidates = {str(k): list(v) for k, v in candidates.items()}
        except Exception:
            return False

        # Validate each chosen string is a valid candidate and matches length
        for sid, chosen in pred.items():
            if sid not in slots or sid not in candidates:
                return False
            if chosen not in candidates[sid]:
                return False
            if len(chosen) != slots[sid]:
                return False

        # Validate all intersection constraints
        for item in intersections:
            if not (isinstance(item, (list, tuple)) and len(item) == 4):
                return False
            u, iu, v, iv = item
            u, v = str(u), str(v)

            try:
                iu, iv = int(iu), int(iv)
            except Exception:
                return False

            if u not in pred or v not in pred:
                return False

            su, sv = pred[u], pred[v]
            if iu < 0 or iv < 0 or iu >= len(su) or iv >= len(sv):
                return False
            if su[iu] != sv[iv]:
                return False

        return True
