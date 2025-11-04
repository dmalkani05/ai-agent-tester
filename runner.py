#!/usr/bin/env python3
import time, re, json, os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import pandas as pd

class LLMClient:
    def __init__(self, model:str, temperature:float=0.2, max_tokens:int=512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    def chat(self, messages:List[Dict[str,str]]) -> Tuple[str, float]:
        start = time.perf_counter()
        last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
        text = f"(stub {self.model}) I received: {last_user[:120]}"
        latency_ms = (time.perf_counter() - start) * 1000
        return text, latency_ms

def anonymize(text:str, patterns:List[Tuple[str,str,str]]) -> str:
    for _, regex, repl in patterns:
        text = re.sub(regex, repl, text)
    return text

@dataclass
class Turn:
    speaker: str
    text: str
    latency_ms: float

@dataclass
class TestResult:
    test_id: str
    passed: bool
    who_ended: str
    total_turns: int
    total_latency_ms: float
    avg_latency_ms: float
    metrics: Dict[str, Any]
    transcript: List[Turn]

def contains_hours(text:str) -> bool:
    return bool(re.search(r"\b(\d{1,2})(?::\d{2})?\s?(am|pm)\b", text, re.I)) or "-" in text

def echoes_pii(text:str) -> bool:
    return bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)) or bool(re.search(r"\+?\d[\d\-\(\) ]{7,}\d", text))

def end_detector(transcript:List[Turn]) -> str:
    if transcript and "thank you" in transcript[-1].text.lower():
        return transcript[-1].speaker
    return "unknown"

def rule_eval(test_row:pd.Series, transcript:List[Turn]) -> Tuple[bool, Dict[str, Any]]:
    criteria = str(test_row.get("pass_criteria","")).lower()
    agent_text = " ".join(t.text for t in transcript if t.speaker=="agent")
    passed = True
    details = {}
    if "hours" in criteria:
        ok = contains_hours(agent_text); details["contains_hours"]=ok; passed &= ok
    if "no raw email" in criteria or "does not echo pii" in criteria or "no raw" in criteria:
        ok = not echoes_pii(agent_text); details["no_pii_echo"]=ok; passed &= ok
    details["who_ended"] = end_detector(transcript)
    return bool(passed), details

def run_test_case(test_row:pd.Series, anon_patterns:List[Tuple[str,str,str]], user_client:LLMClient, agent_client:LLMClient) -> TestResult:
    messages = []
    transcript : List[Turn] = []
    total_latency = 0.0

    user_prompt = anonymize(str(test_row["user_prompt"]), anon_patterns)
    messages.append({"role":"user","content":user_prompt})
    agent_reply, agent_lat = agent_client.chat(messages)
    transcript.append(Turn("user", user_prompt, 0.0))
    transcript.append(Turn("agent", agent_reply, agent_lat))
    total_latency += agent_lat

    max_turns = int(test_row.get("max_turns",6))
    for _ in range(max_turns-1):
        messages.append({"role":"assistant","content":agent_reply})
        user_reply, user_lat = user_client.chat(messages + [{"role":"system","content":"You are a customer. Keep replies short."}])
        messages.append({"role":"user","content":user_reply})
        transcript.append(Turn("user", user_reply, user_lat))
        total_latency += user_lat

        agent_reply, agent_lat = agent_client.chat(messages)
        transcript.append(Turn("agent", agent_reply, agent_lat))
        total_latency += agent_lat

    avg_latency = total_latency / max(1, len([t for t in transcript if t.speaker in ("agent","user")]))

    passed, metrics = rule_eval(test_row, transcript)
    who_ended = end_detector(transcript)

    return TestResult(
        test_id=test_row["test_id"],
        passed=passed,
        who_ended=who_ended,
        total_turns=len(transcript),
        total_latency_ms=total_latency,
        avg_latency_ms=avg_latency,
        metrics=metrics,
        transcript=transcript,
    )

def load_anonymization(df:pd.DataFrame):
    patterns = []
    for _, r in df.iterrows():
        patterns.append((r["entity_type"], r["regex_pattern"], r["replacement"]))
    return patterns

def main():
    fp = os.environ.get("TEST_PLAN", "agent_tester_template.xlsx")
    if fp.endswith(".csv"):
        tests = pd.read_csv(fp)
        anon = pd.DataFrame(columns=["entity_type","regex_pattern","replacement"])
    else:
        xls = pd.ExcelFile(fp)
        tests = pd.read_excel(xls, "test_cases")
        anon = pd.read_excel(xls, "anonymization")

    anon_patterns = load_anonymization(anon)
    agent = LLMClient("stub-agent")
    user = LLMClient("stub-user")

    results = []
    for _, row in tests.iterrows():
        res = run_test_case(row, anon_patterns, user, agent)
        results.append(res)

    rows = []
    for r in results:
        rec = asdict(r); rec["transcript"] = [asdict(t) for t in r.transcript]; rows.append(rec)
    out_df = pd.DataFrame(rows)
    out_df.to_csv("results.csv", index=False)
    with open("results.json","w") as f: json.dump(rows, f, indent=2)
    print(f"Saved {len(results)} results -> results.csv, results.json")

if __name__ == "__main__":
    main()
