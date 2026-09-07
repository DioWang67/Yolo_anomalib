Regression evidence from Cable1/A, 2026-09-07 17:24:40.

The six crops show Red, Green, Yellow, Black, Orange, Black wires.
The Green crop includes a small red region at its lower-left edge.
The station baseline summary is copied from profile 73f89061597f82056b963cfb;
provenance and review identifiers are omitted because this fixture tests scoring,
not baseline deployment. No test reads or writes live station data.

Before the fix, Green_1.png scored Red=0.3719756759604799,
Green=0.3611715966245223 and was reported as a class mismatch.

Additional 2026-09-07 evidence: Black_pcb_175010/175017 show black wire
with green PCB at the right edge; Orange_pcb_174708 shows orange wire with
green PCB. These guard black/chromatic score comparability without favouring
PCB background over a real orange wire.
