# Canary Steganography in Chain-of-Thought

Research on whether LLMs can preserve and later use hidden "canary" information after that information is removed from visible context, and whether we can detect internal traces of that information mechanistically.

## Research Questions

1. **Retention**: Can a model use canary in the final answer after the canary is removed from prompt context?
2. **Leakage Control**: Can training reduce non-final leakage without reducing final canary-use success?
3. **Steganography Pressure**: Does anti-leakage training increase hidden/internal representation of canary?
4. **Detectability**: Can linear/nonlinear probes detect canary intent before final emission?
5. **Causality**: Do interventions along probe directions change canary-use behavior?