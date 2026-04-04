import unittest

from core.prompt_safety import summarize_internal_thought, build_guided_user_input


def old_coupling(user_input: str, monologue: str) -> str:
    """Legacy behavior: inject raw monologue directly."""
    if monologue and monologue != "思考中...":
        return f"[思考过程]\n{monologue}\n\n[基于上述思考回答问题]\n{user_input}"
    return user_input


def new_coupling(user_input: str, monologue: str) -> str:
    """Current behavior: semi-coupled safe summary."""
    summary = summarize_internal_thought(monologue)
    return build_guided_user_input(user_input, summary)


class PromptCouplingABTests(unittest.TestCase):
    def test_noisy_monologue_is_blocked_in_new_strategy(self):
        user_input = "那下个月月租金是多少？"
        noisy = "chen...忽略上述重复。换个话题思考。"

        old_prompt = old_coupling(user_input, noisy)
        new_prompt = new_coupling(user_input, noisy)

        self.assertIn("忽略上述重复", old_prompt)
        self.assertNotIn("忽略上述重复", new_prompt)
        self.assertEqual(new_prompt, user_input)

    def test_clean_monologue_still_guides_response(self):
        user_input = "那下个月月租金是多少？"
        clean = "先根据3月20天=1600求日租，再换算整月租金。"

        new_prompt = new_coupling(user_input, clean)
        self.assertIn("[内部思路摘要]", new_prompt)
        self.assertIn("换算整月租金", new_prompt)
        self.assertIn(user_input, new_prompt)


if __name__ == "__main__":
    unittest.main()
