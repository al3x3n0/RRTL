import unittest

import pyrtl

from rrtl_pyrtl import simulate_lane_trace


class LaneTraceTests(unittest.TestCase):
    def tearDown(self):
        pyrtl.reset_working_block()

    def test_simulate_lane_trace_emits_step_major_lane_vectors(self):
        pyrtl.reset_working_block()
        a = pyrtl.Input(4, "a")
        q = pyrtl.Register(4, "q")
        out = pyrtl.Output(4, "out")
        q.next <<= a
        out <<= q + a
        block = pyrtl.working_block()

        trace = simulate_lane_trace(
            [
                [{"a": 1}, {"a": 4}],
                [{"a": 2}, {"a": 5}],
            ],
            block,
        )

        self.assertEqual(trace["schema"], "rrtl-pyrtl-lane-trace-v1")
        self.assertEqual(trace["lanes"], 2)
        self.assertEqual(trace["steps"][0]["inputs"]["a"], [1, 2])
        self.assertEqual(trace["steps"][0]["outputs"]["out"], [1, 2])
        self.assertEqual(trace["steps"][1]["outputs"]["out"], [5, 7])

    def test_simulate_lane_trace_rejects_uneven_steps(self):
        pyrtl.reset_working_block()
        pyrtl.Input(1, "a")
        with self.assertRaisesRegex(ValueError, "lane 1 has 0 steps"):
            simulate_lane_trace([[{"a": 1}], []], pyrtl.working_block())


if __name__ == "__main__":
    unittest.main()
