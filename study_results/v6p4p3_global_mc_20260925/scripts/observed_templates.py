"""Production interpolation using every verified v16 TC anchor, 60--240 MeV.

Only the allowed anchor set extends the frozen v6.3.7 interpolation algorithm.
At every anchor the template, core location and width equal the direct MC.
The 60--80 MeV acceptance-transition interpolation is a model assumption.
"""
from pathlib import Path
import numpy as np
from archived_templates import TemplateBank
B=Path(__file__).resolve().parents[1]
class ProductionTemplateBank(TemplateBank):
    def __init__(self):
        super().__init__(B)
        self.anchors=np.array([m for m in sorted(self.samples) if 60<=m<=240])
BANK=ProductionTemplateBank()
