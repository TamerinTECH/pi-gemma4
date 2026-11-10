# Calibration texts for expert-routing traces

Small, license-clean corpus used by the expert tracer (`tools/expert-trace`) and the residual-oracle
activation capture. Mixed domains on purpose: routing locality must be measured across topic switches,
scripts (Hebrew vs English), and code.

| File | Content | Origin |
|---|---|---|
| `he_discussion_llm_efficiency.txt` | Long Hebrew technical discussion about efficient LLM inference (the discussion this experiment started from) | conversation notes, 2026-10 |
| `en_tech_moe_streaming.txt` | English technical explainer: MoE routing, expert caching, Raspberry Pi byte budgets | written for this repo |
| `en_narrative_bakery.txt` | English narrative prose (a small-town bakery morning) | written for this repo |
| `code_cpp_llamacpp_eval_callback.cpp` | C++ source | llama.cpp `examples/eval-callback/eval-callback.cpp` (MIT) |
| `code_cpp_llamacpp_debug.cpp` | C++ source | llama.cpp `common/debug.cpp` (MIT) |
| `de_schule_kleincomputer.txt` | German expository prose (small computers in Bavarian schools, privacy, MoE, waiting time, arithmetic) | written for this repo |
| `mixed_qa_numbers.txt` | Short Q&A with arithmetic, dates, lists, a few Hebrew lines | written for this repo |

Concatenate in any order; the tracer records file boundaries so topic switches can be analysed.
