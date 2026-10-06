"""Select the paper implementation explicitly; never infer a legacy prior."""
import argparse


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--backend', default='paper',
                          choices=('paper', 'customized', 'legacy-observable', 'official'),
                          help='paper is the instrumented four-stage FIFO model; '
                               'legacy-observable explicitly selects the historical upstream-only prior')
    options, remaining = selector.parse_known_args(argv)
    if options.backend in ('legacy-observable', 'official'):
        from official import legacy_main
        return legacy_main(remaining)
    from native import main as paper_main
    return paper_main(remaining)
