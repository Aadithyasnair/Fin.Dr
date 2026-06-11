import subprocess
import json
import os

from config import (
    OLLAMA_EXE,
    OLLAMA_MODEL
)


class LocalLLM:

    def __init__(self):

        if not os.path.exists(OLLAMA_EXE):
            raise RuntimeError(
                f"Ollama not found:\n{OLLAMA_EXE}"
            )

    def generate(self, prompt):

        try:

            result = subprocess.run(
                [
                    OLLAMA_EXE,
                    "run",
                    OLLAMA_MODEL,
                    prompt
                ],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                raise Exception(result.stderr)

            return result.stdout.strip()

        except Exception as e:

            return (
                "Unable to generate explanation. "
                f"Error: {e}"
            )


llm = LocalLLM()