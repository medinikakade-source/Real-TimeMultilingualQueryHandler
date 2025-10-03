# backend/llm_stub.py
from typing import List, Dict

class LLMStub:
    """
    Small deterministic stub for an LLM provider.
    - `generate(prompt)` returns a composed answer and metadata.
    - Meant for dev/demo only. Replace with real provider client later.
    """

    def __init__(self, name: str = "LLM-Stub", max_output_chars: int = 800):
        self.name = name
        self.max_output_chars = max_output_chars

    def generate(self, prompt: str) -> Dict:
        """
        Very simple deterministic 'generation'.
        Strategy:
          - Look for 'Question:' line and respond with a short summary
          - Include the top sources referenced in the prompt (if present)
          - If prompt contains 'NO_RELEVANT_INFO' marker, return a safe "I don't know" answer
        """
        # Basic heuristics to simulate an LLM output
        q_marker = "Question:"
        src_marker = "CONTEXT:\n"
        answer_text = ""
        # If the user asked for no-assumptions, check prompt marker
        if "NO_RELEVANT_INFO" in prompt:
            answer_text = "I don't have enough information in the provided sources to answer that confidently."
        else:
            # Attempt to extract the question line
            question = None
            for line in prompt.splitlines():
                if line.strip().startswith(q_marker):
                    question = line.split(q_marker, 1)[1].strip()
                    break
            # Grab context snippet list (simple parse)
            contexts = []
            if src_marker in prompt:
                contexts_block = prompt.split(src_marker, 1)[1]
                # contexts may be a series of '--- source: ...\n<text>\n'
                parts = contexts_block.split("---")
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    # try to parse 'source: ' header
                    lines = p.splitlines()
                    header = lines[0] if lines else ""
                    body = " ".join(lines[1:])[:400] if len(lines) > 1 else ""
                    contexts.append({"header": header, "body": body})

            # Build a concise "answer" using found info
            if question is None:
                answer_text = "I could not find the question in the prompt."
            else:
                if contexts:
                    # echo a short synthetic answer that references the first context
                    first = contexts[0]
                    answer_text = f"Based on the provided sources (see {first.get('header', 'source')}), the short answer is: {first.get('body','').split('.')[0]}. If you want more details, ask follow-ups."
                else:
                    # fallback: safe unknown
                    answer_text = "I couldn't find content relevant to that question in the provided contexts."

        # truncate
        answer_text = answer_text.strip()
        if len(answer_text) > self.max_output_chars:
            answer_text = answer_text[: self.max_output_chars - 3] + "..."

        # Provide a small metadata dict like a provider would
        metadata = {
            "provider": self.name,
            "length": len(answer_text),
            "note": "This is a deterministic stub — replace with real LLM for production."
        }
        return {"answer": answer_text, "metadata": metadata}
