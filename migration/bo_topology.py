"""Physical GPU mappings for the native P2P AFD BO protocol (stdlib only)."""
import re


REFERENCES = {4: (2, 2), 6: (4, 2), 8: (4, 4)}


def validate_counts(attention, expert):
    if (attention < 1 or expert < 1 or attention + expert > 8
            or attention < expert or attention % expert):
        raise ValueError('P2P AFD requires A >= E, A divisible by E, and at most eight active GPUs')


def parse_topology(name):
    match = re.fullmatch(r'([1-9][0-9]*)a([1-9][0-9]*)e', name.lower())
    if not match:
        raise ValueError('Invalid A/E topology name')
    counts = tuple(map(int, match.groups()))
    validate_counts(*counts)
    return counts


def topologies(gpus, num_experts=None):
    """Enumerate every legal A/E size up to the allocation, reference first.

    Physical mappings use a fixed prefix of the supplied GPU order; permutations
    of otherwise identical cards are not additional candidates. Model divisibility
    is optional here so host-only preflight can validate its reference mapping.
    """
    if (len(gpus) not in REFERENCES or len(set(gpus)) != len(gpus)
            or any(type(g) is not int or g < 0 for g in gpus)):
        raise ValueError('Reserve 4, 6, or 8 distinct nonnegative physical GPU indices')
    if num_experts is not None and (type(num_experts) is not int or num_experts < 1):
        raise ValueError('Model expert count must be a positive integer')
    counts = [(na, ne) for ne in range(1, len(gpus)) for na in range(ne, len(gpus) - ne + 1)
              if na % ne == 0 and (num_experts is None or num_experts % ne == 0)]
    reference = REFERENCES[len(gpus)]
    if reference not in counts:
        raise ValueError('Model expert count cannot deploy the reference topology')
    counts.remove(reference)
    counts.insert(0, reference)
    return [dict(id=f'{na}a{ne}e', attention_gpus=gpus[:na], expert_gpus=gpus[na:na + ne],
                 attention_dp=na, attention_tp=1, expert_dp=1, expert_ep=ne, expert_tp=1)
            for na, ne in counts]


def validate_groups(attention, expert, allocation):
    validate_counts(len(attention), len(expert))
    active = attention + expert
    if (len(set(active)) != len(active) or len(set(allocation)) != len(allocation)
            or not set(active) <= set(allocation) or any(g < 0 for g in allocation)):
        raise ValueError('Invalid or overlapping physical GPU mapping')
