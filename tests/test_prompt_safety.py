import unittest

from core.prompt_safety import summarize_internal_thought, build_guided_user_input


class PromptSafetyTests(unittest.TestCase):
    def test_filters_loop_breaker_monologue(self):
        text = "chenchen……忽略上述重复。换个话题思考。"
        self.assertEqual(summarize_internal_thought(text), "")

    def test_keeps_clean_thought_summary(self):
        text = "先提取已知条件，再按比例计算下个月月租。最后核对押金与卫生费不计入月租。"
        summary = summarize_internal_thought(text)
        self.assertIn("提取已知条件", summary)
        self.assertTrue(len(summary) <= 60)

    def test_build_guided_input(self):
        user_input = "那下个月月租多少？"
        guided = build_guided_user_input(user_input, "先列条件再计算")
        self.assertIn("[内部思路摘要]", guided)
        self.assertIn(user_input, guided)


if __name__ == "__main__":
    unittest.main()
