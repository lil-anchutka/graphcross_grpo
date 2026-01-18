
from torch.utils.data import Dataset
import json
import random
from typing import List, Optional, Dict, Any

DEFAULT_SYSTEM_PROMPT = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
""".strip()

def to_chat_sample(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    question = sample["question"]
    answer = sample["answer"]
    out = {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "answer": answer,
    }
    if "difficulty" in sample:
        out["difficulty"] = sample["difficulty"]
    if "metadata" in sample:
        out["metadata"] = sample["metadata"]

    return out


class GraphCrossEvalDataset(Dataset):
    def __init__(self, jsonl_path: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                self.samples.append(to_chat_sample(raw, self.system_prompt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GraphCrossTrainDataset(Dataset):
    """
    Fixed (pre-sampled) train dataset. Compatible with GRPOTrainer.
    """
    def __init__(
        self,
        env,
        difficulties: List[int],
        n_samples: int,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        seed: Optional[int] = 42,
        max_attempts_per_item: int = 50,
        gen_max_attempts: int = 200,
    ):
        self.env = env
        self.difficulties = list(difficulties)
        self.n_samples = int(n_samples)
        self.system_prompt = system_prompt
        self.seed = seed
        self.max_attempts_per_item = int(max_attempts_per_item)
        self.gen_max_attempts = int(gen_max_attempts)

        rng = random.Random(seed)
        self.samples: List[Dict[str, Any]] = []

        while len(self.samples) < self.n_samples:
            diff = rng.choice(self.difficulties)

            raw = None
            for _ in range(self.max_attempts_per_item):
                batch = self.env.generate(
                    num_of_questions=1,
                    max_attempts=self.gen_max_attempts,
                    difficulty=diff,
                )
                if batch:
                    raw = batch[0].to_json()
                    break

            if raw is None:
                continue

            self.samples.append(to_chat_sample(raw, self.system_prompt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
