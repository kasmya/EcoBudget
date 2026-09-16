"""Phase A: end-to-end energy accounting -- COMPUTE (running the models) vs
TRANSFER (moving bytes over a 5G access path) -- so we can report NET energy
per query and answer the existential question: do the bytes saved by stopping
early outweigh the joules spent running flan-t5 + RoBERTa-QA + MiniLM?

Everything here is a first-order engineering estimate with explicitly cited,
configurable coefficients. Nothing is measured on real hardware yet (that is a
later phase); the goal is a defensible order-of-magnitude comparison and a
model that is 5G-grounded rather than a single generic web-CO2 coefficient.

Two independent models:
  * ComputeModel: transformer inference energy = FLOPs / (device FLOPs-per-joule).
    FLOPs ~= 2 * params * (input_tokens + output_tokens) per forward pass (the
    standard 2xMAC approximation), summed over every model call in a query.
  * FiveGTransferModel: energy per transported bit split into RAN (access) +
    transport (core) + device modem receive -- the RAN dominates 5G energy.

Carbon: joules -> kWh -> gCO2e via grid carbon intensity.

Coefficient sources are documented in docs/energy_model.md; the defaults below
are mid-range literature values, labelled as assumptions, and every one is a
constructor argument so a reviewer can swap them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Model sizes (trainable+frozen params, approximate) -------------------
MODEL_PARAMS = {
    "flan-t5-small": 77_000_000,     # decomposer-small / answer-small base
    "flan-t5-base": 248_000_000,     # decomposer-base-lora / answer-base base
    "minilm": 22_700_000,            # all-MiniLM-L6-v2 query embedding
    "roberta-base-squad2": 125_000_000,  # QA evidence scorer
}

# Average token counts per op type (documented approximations; see energy_model.md).
# input+output tokens processed by one call of that op.
OP_TOKENS = {
    "decompose": (24, 24),       # flan-t5-base: question in, entity|attribute out
    "query_embed": (10, 0),      # MiniLM: one requirement query encoded
    "qa_score": (96, 8),         # roberta: question+passage in, short span out
    "answer_gen": (160, 24),     # flan-t5-base: prompt(question+evidence) in, answer out
}

OP_MODEL = {
    "decompose": "flan-t5-base",
    "query_embed": "minilm",
    "qa_score": "roberta-base-squad2",
    "answer_gen": "flan-t5-base",
}


@dataclass
class OpCounts:
    """How many times each model op runs for a single query (logical count,
    independent of any caching used to speed up the experiment)."""
    decompose: int = 0
    query_embed: int = 0
    qa_score: int = 0
    answer_gen: int = 0

    def as_dict(self) -> dict:
        return {"decompose": self.decompose, "query_embed": self.query_embed,
                "qa_score": self.qa_score, "answer_gen": self.answer_gen}


@dataclass
class ComputeModel:
    """Inference energy from FLOPs and a device-class efficiency.

    `flops_per_joule` is the dominant assumption. Defaults to an edge-CPU class
    (~50 GFLOPS/W = 5e10 FLOPS/J). Mobile-SoC NPUs are far more efficient
    (1-10 TOPS/W); a server GPU is different again -- swap per target device."""
    flops_per_joule: float = 5.0e10  # edge-CPU class assumption

    def _flops(self, op: str, count: int) -> float:
        if count == 0:
            return 0.0
        params = MODEL_PARAMS[OP_MODEL[op]]
        tin, tout = OP_TOKENS[op]
        # 2 * params * tokens per forward pass (2xMAC); generation counts
        # output tokens as additional autoregressive steps.
        return count * 2.0 * params * (tin + tout)

    def energy_joules(self, ops: OpCounts) -> float:
        total_flops = sum(self._flops(op, getattr(ops, op)) for op in OP_TOKENS)
        return total_flops / self.flops_per_joule

    def breakdown_joules(self, ops: OpCounts) -> dict:
        return {op: self._flops(op, getattr(ops, op)) / self.flops_per_joule
                for op in OP_TOKENS}


@dataclass
class FiveGTransferModel:
    """Energy to move one bit over a 5G access path, split RAN + transport +
    device-modem receive. RAN dominates mobile-network energy (~70-80%).

    Defaults derive from a mid-range mobile-access figure of ~0.05 kWh/GB for
    the 5G era (studies span ~0.01-0.1 kWh/GB); that is 0.05*3.6e6/8e9 =
    2.25e-8... note: 0.05 kWh/GB = 1.8e5 J / 8e9 bit = 2.25e-5 J/bit network,
    split 75% RAN / 25% transport, plus a separate device-modem receive term."""
    network_j_per_bit: float = 2.25e-5   # RAN + transport (mobile access, 5G-era assumption)
    ran_fraction: float = 0.75           # RAN share of network energy
    device_rx_j_per_bit: float = 4.0e-7  # smartphone modem receive energy per bit (assumption)

    @property
    def ran_j_per_bit(self) -> float:
        return self.network_j_per_bit * self.ran_fraction

    @property
    def transport_j_per_bit(self) -> float:
        return self.network_j_per_bit * (1.0 - self.ran_fraction)

    def energy_joules(self, num_bytes: int) -> float:
        bits = num_bytes * 8
        return bits * (self.network_j_per_bit + self.device_rx_j_per_bit)

    def breakdown_joules(self, num_bytes: int) -> dict:
        bits = num_bytes * 8
        return {"ran": bits * self.ran_j_per_bit,
                "transport": bits * self.transport_j_per_bit,
                "device_rx": bits * self.device_rx_j_per_bit}


@dataclass
class PayloadModel:
    """Maps a set of retrieved passages to the bytes actually transported over
    the link, under a chosen realism scenario (Phase B).

    Phase A used raw extracted-text bytes (~25-259 B) -- an under-count, because
    on a real link you fetch a web resource/page, not a clean snippet. Scenarios
    span the realistic range so the compute-vs-transfer finding is reported
    across assumptions rather than pinned to one:
      - "text":      extracted text bytes (Phase A; optimistic lower bound).
      - "resource":  the passage as a real web resource -- text inflated for
                     HTML markup + HTTP/TLS overhead, with a floor.
      - "html_page": fetch the source HTML document once per unique page
                     (HTTP-Archive-class median HTML weight).
      - "full_page": fetch the full page with assets once per unique page
                     (HTTP-Archive-class median total page weight).
    Page scenarios de-duplicate by source_url (one fetch serves all its
    passages), which is what rewards a policy that touches fewer pages.
    """
    resource_inflation: float = 4.0        # HTML markup + overhead vs extracted text
    resource_floor_bytes: int = 800        # min realistic resource-on-wire size
    html_page_bytes: int = 60_000          # median content-page HTML document (assumption)
    full_page_bytes: int = 2_000_000       # median full page weight with assets (assumption)

    def transfer_bytes(self, passages, scenario: str = "text") -> int:
        if scenario == "text":
            return sum(p.byte_size for p in passages)
        if scenario == "resource":
            return sum(max(int(p.byte_size * self.resource_inflation), self.resource_floor_bytes)
                       for p in passages)
        # page-level: one fetch per unique source
        sources = {getattr(p, "source_url", "") or p.passage_id for p in passages}
        n_pages = len(sources)
        if scenario == "html_page":
            return n_pages * self.html_page_bytes
        if scenario == "full_page":
            return n_pages * self.full_page_bytes
        raise ValueError(f"unknown scenario: {scenario!r}")


SCENARIOS = ("text", "resource", "html_page", "full_page")


@dataclass
class EnergyAccountant:
    """Combines compute + 5G transfer into total joules and gCO2e."""
    compute: ComputeModel = field(default_factory=ComputeModel)
    transfer: FiveGTransferModel = field(default_factory=FiveGTransferModel)
    payload: PayloadModel = field(default_factory=PayloadModel)
    grid_g_per_kwh: float = 475.0  # global-average grid carbon intensity (IEA-class assumption)

    def account_passages(self, ops: OpCounts, passages, scenario: str = "text") -> dict:
        """Account energy with transfer bytes derived from the retrieved
        passages under a payload realism scenario (Phase B)."""
        return self.account(ops, self.payload.transfer_bytes(passages, scenario))

    def account(self, ops: OpCounts, num_bytes: int) -> dict:
        compute_j = self.compute.energy_joules(ops)
        transfer_j = self.transfer.energy_joules(num_bytes)
        total_j = compute_j + transfer_j
        gco2e = (total_j / 3.6e6) * self.grid_g_per_kwh  # J -> kWh -> gCO2e
        return {
            "compute_j": compute_j,
            "transfer_j": transfer_j,
            "total_j": total_j,
            "compute_fraction": compute_j / total_j if total_j else 0.0,
            "gco2e": gco2e,
        }


# --- Logical op-counting for the pipeline (caching-independent) ------------

def query_op_counts(n_requirements: int, n_passages_added: int,
                    answer_mode: str = "per_requirement") -> OpCounts:
    """The model ops one query performs, independent of any experiment cache:
      - 1 decompose (question -> requirements)
      - 1 query embedding per requirement (retrieval)
      - 1 QA score per (requirement, added passage): the tracker scores every
        added passage against every requirement
      - answer generations: 1 per requirement (per_requirement) or 1 (joint)
    """
    n_gen = n_requirements if answer_mode == "per_requirement" else 1
    return OpCounts(
        decompose=1,
        query_embed=n_requirements,
        qa_score=n_requirements * n_passages_added,
        answer_gen=n_gen,
    )
