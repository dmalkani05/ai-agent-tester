
# AI Agent Tester (Starter Harness)

**What this is:** A minimal, pluggable harness to test conversational AI agents using **agent-to-agent** simulations, with **anonymization**, **latency** measurements, and **rule/eval**-based pass criteria.

## How to use
1. Open `agent_tester_template.xlsx` and add/modify rows in `test_cases`.
2. (Optional) Add regex rules to `anonymization` to scrub PII.
3. Run the harness (stubbed models):
   ```bash
   TEST_PLAN=agent_tester_template.xlsx python3 runner.py
   ```
4. Check `results.csv` and `results.json` for pass/fail, metrics, and transcripts.

## Plug in real providers
- Replace `LLMClient.chat` with your real model API calls (OpenAI/Anthropic/etc.).
- Add TTS/ASR if your agent expects audio I/O.
- Implement an `eval-LLM` function to grade transcripts against `rubric`.

## Test case columns
`test_id, scenario_name, user_prompt, persona, expected_outcome, pass_criteria, end_condition_expected, max_turns, eval_type, tags`
