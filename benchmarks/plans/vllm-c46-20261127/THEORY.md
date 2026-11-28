# Theory and evidence

The local vLLM recipe has full decode-only CUDA graphs and an optional CPU host-expert runtime. The strongest measured advantage on this workload came from enabling CPU experts: vLLM was 52–76 aggregate tok/s faster with CPU experts on than off, while its GPU-only comparison was slower than the recent native control. This points to host/GPU work overlap as the best measured lead source. Native CPU experts previously regressed on six 404k contexts, so any forward port must be measured against the short coding edit and full-context constraints.

vLLM also has an adaptive DSpark verifier that jointly allocates proposal positions by cumulative survival probability and a profiled step-cost model. The local recipe does not enable this option. A native fixed-cap prototype was tested but rejected: a four-token/request cap lost 3.1–3.5%, and a six-token cap lost 7.9–8.2%. A production-quality port needs cost calibration and GPU-side allocation rather than a static cap.
