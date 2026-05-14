"""Orchestrator: select_and_label → count_triangles → render_bench → analyze."""
import time

from . import analyze, count_triangles, render_bench, select_and_label


def main():
    t_start = time.perf_counter()

    print("\n========== STEP 1+2: select + label_mesh ==========")
    select_and_label.main()

    print("\n========== STEP 4: count triangles ==========")
    count_triangles.main()

    print("\n========== STEP 3: render benchmark ==========")
    render_bench.main()

    print("\n========== STEP 5: regression + report ==========")
    analyze.main()

    print(f"\n========== DONE in {time.perf_counter() - t_start:.1f}s ==========")


if __name__ == "__main__":
    main()
