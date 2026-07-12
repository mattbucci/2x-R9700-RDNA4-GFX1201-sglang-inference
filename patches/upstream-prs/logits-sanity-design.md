# Logits finite-value diagnostics

## Problem

A NaN or infinite logit row can surface later as an asynchronous multinomial or device assertion, obscuring the originating request and layer.

## Proposal

Add an opt-in sampler setting:

\`\`\`text
--logits-sanity off|warn|raise
\`\`\`

Use one finite-value reduction per row. In \`warn\` mode, log the request and row and replace the invalid row with a safe distribution. In \`raise\` mode, raise a synchronous error containing the row index. Keep the default \`off\`.

The implementation should cover NaN and both infinities, include unit tests for all modes, and report measured overhead at M=1.
