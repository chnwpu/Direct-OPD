import unittest

import numpy as np

from grpo_dynamics.dynamics import shift, student_probe, teacher
from grpo_dynamics.prepare import PROTOCOL, create_splits
from grpo_dynamics.record import sampling_seeds
from grpo_dynamics.verifier import score_answer


class ProtocolTests(unittest.TestCase):
    def test_binary_verifier(self):
        cases = [
            (r"worked out \boxed{\frac{1}{2}}", "0.5", 1),
            (r"\boxed{x^2+2x+1}", "(x+1)^2", 1),
            ("Final answer: -3", "-3", 1),
            (r"\boxed{2} then \boxed{4}", "2", 0),
            ("We considered 10, but did not finish.", "10", 0),
            (r"incomplete \boxed{1", "1", 0),
            ("", "0", 0),
            (r"\boxed{0}", "0", 1),
        ]
        for ans, gold, want in cases:
            with self.subTest(ans=ans, gold=gold):
                self.assertEqual(score_answer(ans, gold)["score"], want)

    def test_deterministic_split_and_overlap(self):
        raw = [{"question": f"Problem {i}", "final_answer": "1", "difficulty": 5, "topic": "test"}
               for i in range(2000)]
        raw += [{**raw[0], "question": "  PROBLEM   0 "}]
        result, _ = create_splits(raw)
        reverse, _ = create_splits(reversed(raw))
        self.assertEqual(result, reverse)
        sets = [set(r["extra_info"]["problem_id"] for r in result[k])
                for k in ("train","validation","probe","test")]
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
        self.assertEqual(len(result["preflight"]), PROTOCOL["preflight_size"])

    def test_seed_identity(self):
        a, b = sampling_seeds(1, 1024), sampling_seeds(2, 1024)
        self.assertEqual(len(set(a)), 1024)
        self.assertFalse(set(a) & set(b))
        np.testing.assert_array_equal(sampling_seeds(0, 32, True), sampling_seeds(10, 32, True))

    def test_arbitrary_shift_and_kl(self):
        logp = np.log(np.array([[[.8,.2]],[[.6,.4]],[[.4,.6]]]))
        d = shift(logp,0,2)
        z = teacher(logp,0,2,True,rewards=np.array([.1,.2,.3]))
        self.assertAlmostEqual(z["checkpoint_reward_gain"],.2)
        self.assertAlmostEqual(z["teacher_kl_new_to_old"], .4*np.log(.4/.8)+.6*np.log(.6/.2))
        self.assertAlmostEqual(z["shift_rms"], np.sqrt(np.mean(d*d)))
        s = student_probe(logp,0,1,logp[0],comparison=(1,2))
        self.assertAlmostEqual(s["student_weighted_centered_cosine"],1)
        self.assertIsNone(student_probe(logp,0,1,logp[0]+0,comparison=(0,1)).get("reward_shift_covariance"))

    def test_aligned_trajectory_covariance(self):
        logp = np.log(np.array([[[.5,.5],[.5,.5]],[[.8,.2],[.2,.8]]]))
        extras = dict(observed_index=np.array([0,0]), trajectory_id=np.array([0,1]), trajectory_reward=np.array([1,0]))
        self.assertAlmostEqual(teacher(logp,0,1,True,**extras)["reward_shift_alignment"],np.log(4))
        self.assertAlmostEqual(student_probe(logp,0,1,logp[0],**extras)["reward_shift_covariance"],np.log(4)/4)


if __name__ == "__main__":
    unittest.main()
