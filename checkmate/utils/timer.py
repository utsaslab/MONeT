from decimal import Decimal
from math import ceil, log10
from timeit import default_timer


class Timer:
    def __init__(self, name, extra_data=None, print_results=False, niters=None):
        self.niters = niters
        self._elapsed = Decimal()
        self._name = name
        if extra_data:
            self._name += "; " + str(extra_data)
        self._print_results = print_results
        self._start_time = None
        self._children = {}
        self._count = 0

    @property
    def elapsed(self):
        return float(self._elapsed)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        if self._print_results:
            self.print_results()

    def child(self, name):
        try:
            return self._children[name]
        except KeyError:
            result = Timer(name, print_results=False)
            self._children[name] = result
            return result

    def start(self):
        self._count += 1
        self._start_time = self._get_time()

    def stop(self):
        self._elapsed += self._get_time() - self._start_time

    def print_results(self):
        print(self._format_results())

    def _format_results(self, indent="  "):
        children = self._children.values()
        elapsed = self._elapsed or sum(c._elapsed for c in children)
        result = "%s: %.3fs" % (self._name, elapsed)
        max_count = max(c._count for c in children) if children else 0
        count_digits = 0 if max_count <= 1 else int(ceil(log10(max_count + 1)))
        for child in sorted(children, key=lambda c: c._elapsed, reverse=True):
            lines = child._format_results(indent).split("\n")
            child_percent = child._elapsed / elapsed * 100
            lines[0] += " (%d%%)" % child_percent
            if count_digits:
                # `+2` for the 'x' and the space ' ' after it:
                lines[0] = ("%dx " % child._count).rjust(count_digits + 2) + lines[0]
            for line in lines:
                result += "\n" + indent + line
        return result

    @staticmethod
    def _get_time():
        return Decimal(default_timer())