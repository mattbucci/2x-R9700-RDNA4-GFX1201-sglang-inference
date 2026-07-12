# Upstream PR drafts

This directory holds small changes intended for submission to SGLang rather than permanent repository-only behavior. Drafts are snapshots and must be rebased onto current SGLang \`main\` before submission.

| Area | Local source | Upstream disposition |
|---|---|---|
| Triton attention FP32 accumulation | patch 011 | Re-audit current attention kernels and retain only missing FP32 accumulation sites. |
| Mistral tool-marker omission recovery | patch 040 | Rebase onto the current compact Mistral parser and submit with streaming tests. |
| Logits finite-value diagnostics | design document | New opt-in feature; prior sampler flag no longer exists upstream. |
| MistralCommonBackend opt-out | patch 083 | Candidate tokenizer correctness fix for render-then-encode chat paths. |

Before opening a PR:

1. reproduce the failure on current upstream;
2. reduce the change to the smallest backend-independent form;
3. add unit coverage;
4. test on the affected backend and one unaffected backend;
5. remove repository-specific environment variables and comments.
