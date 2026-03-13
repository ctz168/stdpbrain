"""
独白引擎核心逻辑测试（不依赖 PyTorch）

测试内容:
1. 思维状态机转换
2. 情绪状态更新
3. 主题锚定机制
4. 联想链生成
5. Prompt 构建
"""

import random
import time
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional, Tuple, Dict


# ==================== 复制核心类 ====================

class ThoughtState(Enum):
    FOCUSED = "focused"
    WANDERING = "wandering"
    REFLECTING = "reflecting"
    RESTING = "resting"


class EmotionState(Enum):
    CURIOUS = "curious"
    CONFUSED = "confused"
    EXCITED = "excited"
    CALM = "calm"
    THOUGHTFUL = "thoughtful"


@dataclass
class ThoughtTheme:
    content: str
    keywords: List[str]
    importance: float
    created_time: float
    last_active_time: float
    drift_count: int = 0
    return_count: int = 0


@dataclass
class AssociationLink:
    from_concept: str
    to_concept: str
    link_type: str
    strength: float
    timestamp: float


@dataclass
class MonologueSegment:
    content: str
    theme: Optional[str]
    thought_state: ThoughtState
    emotion: EmotionState
    timestamp: float
    is_reflection: bool = False
    association_from: Optional[str] = None


# ==================== 测试类 ====================

class MonologueEngineTest:
    """独白引擎测试版本（无模型依赖）"""
    
    def __init__(self):
        # 思维状态
        self.current_thought_state = ThoughtState.RESTING
        self.current_emotion = EmotionState.CALM
        self.thought_state_duration = 0
        
        # 主题系统
        self.current_theme: Optional[ThoughtTheme] = None
        self.theme_history: deque = deque(maxlen=5)
        
        # 独白历史
        self.monologue_history: deque = deque(maxlen=20)
        self.monologue_segments: List[MonologueSegment] = []
        
        # 联想链
        self.association_chain: List[AssociationLink] = []
        self.current_concept: str = ""
        
        # 思维种子
        self.thought_seed: str = ""
        self.seed_timestamp: float = 0
        
        # 状态转换概率
        self.state_transition_probs = {
            ThoughtState.FOCUSED: {
                ThoughtState.WANDERING: 0.15,
                ThoughtState.REFLECTING: 0.10,
                ThoughtState.FOCUSED: 0.75
            },
            ThoughtState.WANDERING: {
                ThoughtState.FOCUSED: 0.30,
                ThoughtState.REFLECTING: 0.15,
                ThoughtState.WANDERING: 0.55
            },
            ThoughtState.REFLECTING: {
                ThoughtState.FOCUSED: 0.40,
                ThoughtState.WANDERING: 0.10,
                ThoughtState.RESTING: 0.20,
                ThoughtState.REFLECTING: 0.30
            },
            ThoughtState.RESTING: {
                ThoughtState.FOCUSED: 0.50,
                ThoughtState.WANDERING: 0.30,
                ThoughtState.RESTING: 0.20
            }
        }
        
        # 情绪-风格映射
        self.emotion_style_prompts = {
            EmotionState.CURIOUS: "好奇地探索...",
            EmotionState.CONFUSED: "困惑地思考...",
            EmotionState.EXCITED: "兴奋地联想...",
            EmotionState.CALM: "平静地沉思...",
            EmotionState.THOUGHTFUL: "深沉地反思..."
        }
        
        # 思维状态-风格映射
        self.state_style_prompts = {
            ThoughtState.FOCUSED: "专注于",
            ThoughtState.WANDERING: "思绪飘向",
            ThoughtState.REFLECTING: "反思刚才的想法",
            ThoughtState.RESTING: "静静地"
        }
        
        self._initialize_thought_system()
    
    def _initialize_thought_system(self):
        """初始化思维系统"""
        initial_themes = [
            "存在的意义",
            "思维的奥秘",
            "记忆与时间",
            "自我与他者",
            "知识的边界"
        ]
        theme_content = random.choice(initial_themes)
        self._set_new_theme(theme_content)
        self.current_emotion = EmotionState.CALM
        self.current_concept = "思考"
    
    def _set_new_theme(self, theme_content: str, importance: float = 0.5):
        """设置新的思维主题"""
        now = time.time()
        keywords = self._extract_keywords(theme_content)
        
        self.current_theme = ThoughtTheme(
            content=theme_content,
            keywords=keywords,
            importance=importance,
            created_time=now,
            last_active_time=now
        )
        
        if self.current_theme:
            self.theme_history.append(self.current_theme)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        words = text.replace("的", " ").replace("与", " ").split()
        return [w for w in words if len(w) > 1][:5]
    
    def _transition_thought_state(self):
        """思维状态转换"""
        current = self.current_thought_state
        probs = self.state_transition_probs[current]
        
        rand = random.random()
        cumulative = 0
        for next_state, prob in probs.items():
            cumulative += prob
            if rand < cumulative:
                self.current_thought_state = next_state
                self.thought_state_duration = 0
                return
        
        self.thought_state_duration += 1
    
    def _update_emotion_state(self, context: str = ""):
        """更新情绪状态"""
        if "？" in context or "困惑" in context or "不明白" in context:
            self.current_emotion = EmotionState.CONFUSED
        elif "！" in context or "发现" in context or "原来" in context:
            self.current_emotion = EmotionState.EXCITED
        elif "思考" in context or "反思" in context:
            self.current_emotion = EmotionState.THOUGHTFUL
        elif "为什么" in context or "如何" in context:
            self.current_emotion = EmotionState.CURIOUS
        else:
            if random.random() < 0.2:
                emotions = list(EmotionState)
                self.current_emotion = random.choice(emotions)
    
    def _generate_association(self, current_concept: str) -> Tuple[str, str]:
        """生成联想"""
        link_types = ["similarity", "causality", "contrast", "temporal"]
        link_type = random.choice(link_types)
        
        associations = {
            "思考": ["记忆", "意识", "存在"],
            "记忆": ["时间", "遗忘", "身份"],
            "时间": ["变化", "永恒", "瞬间"],
            "存在": ["虚无", "意义", "自我"],
            "自我": ["他者", "关系", "认知"],
            "意识": ["觉知", "思维", "存在"],
            "意义": ["价值", "目的", "创造"]
        }
        targets = associations.get(current_concept, ["思考", "记忆", "存在"])
        target = random.choice(targets)
        
        return target, link_type
    
    def _should_return_to_theme(self) -> bool:
        """判断是否应该回归主题"""
        if not self.current_theme:
            return False
        if self.current_theme.drift_count > 3:
            return True
        time_since_active = time.time() - self.current_theme.last_active_time
        if time_since_active > 30:
            return True
        return False
    
    def _build_human_like_prompt(self) -> str:
        """构建类人脑独白Prompt"""
        parts = []
        
        # 1. 状态和情绪前缀
        state_prefix = self.state_style_prompts[self.current_thought_state]
        emotion_prefix = self.emotion_style_prompts[self.current_emotion]
        
        # 2. 主题锚定
        if self.current_theme and self.current_thought_state == ThoughtState.FOCUSED:
            parts.append(f"关于「{self.current_theme.content}」")
        
        # 3. 联想链
        if self.current_concept and self.current_thought_state == ThoughtState.WANDERING:
            next_concept, link_type = self._generate_association(self.current_concept)
            if link_type == "similarity":
                parts.append(f"从{self.current_concept}联想到相似的{next_concept}")
            elif link_type == "contrast":
                parts.append(f"想到{self.current_concept}，却又想到相反的{next_concept}")
            elif link_type == "causality":
                parts.append(f"{self.current_concept}让我想到{next_concept}")
            else:
                parts.append(f"从{self.current_concept}到{next_concept}")
            self.current_concept = next_concept
        
        # 4. 反思模式
        if self.current_thought_state == ThoughtState.REFLECTING:
            if self.monologue_history:
                last_thought = list(self.monologue_history)[-1] if self.monologue_history else ""
                parts.append(f"反思刚才的想法：「{last_thought[:30]}...」")
            parts.append("我在想...")
        
        # 5. 回归主题
        if self._should_return_to_theme() and self.current_theme:
            parts.append(f"回到「{self.current_theme.content}」")
            self.current_theme.return_count += 1
            self.current_theme.drift_count = 0
        
        # 6. 思维种子
        if self.thought_seed and time.time() - self.seed_timestamp < 60:
            parts.append(f"用户说：{self.thought_seed[:20]}")
        
        # 7. 历史上下文
        if self.monologue_history and self.current_thought_state != ThoughtState.REFLECTING:
            recent = list(self.monologue_history)[-1] if self.monologue_history else ""
            if recent and len(recent) > 10:
                parts.append(f"刚才想到{recent[-30:]}")
        
        # 8. 构建最终Prompt
        if parts:
            prompt = "，".join(parts) + "..."
        else:
            default_prompts = [
                "此刻我在想...",
                "忽然想到...",
                "思维在流动...",
                "静静地思考...",
                "回忆起..."
            ]
            prompt = random.choice(default_prompts)
        
        return prompt
    
    def simulate_monologue(self, external_stimulus: str = "") -> dict:
        """模拟独白生成"""
        # 处理外部刺激
        if external_stimulus:
            self.thought_seed = external_stimulus
            self.seed_timestamp = time.time()
            if len(external_stimulus) > 5:
                self._set_new_theme(external_stimulus[:30], importance=0.8)
            self.current_thought_state = ThoughtState.FOCUSED
            self.current_emotion = EmotionState.CURIOUS
        
        # 状态转换
        old_state = self.current_thought_state
        self._transition_thought_state()
        new_state = self.current_thought_state
        
        # 更新情绪
        self._update_emotion_state(external_stimulus)
        
        # 构建Prompt
        prompt = self._build_human_like_prompt()
        
        # 记录
        segment = MonologueSegment(
            content=prompt,
            theme=self.current_theme.content if self.current_theme else None,
            thought_state=self.current_thought_state,
            emotion=self.current_emotion,
            timestamp=time.time()
        )
        self.monologue_segments.append(segment)
        self.monologue_history.append(prompt)
        
        # 更新主题漂移计数
        if self.current_theme and self.current_thought_state == ThoughtState.WANDERING:
            self.current_theme.drift_count += 1
        
        return {
            'prompt': prompt,
            'old_state': old_state.value,
            'new_state': new_state.value,
            'emotion': self.current_emotion.value,
            'theme': self.current_theme.content if self.current_theme else None,
            'concept': self.current_concept
        }


# ==================== 运行测试 ====================

def run_tests():
    print("=" * 60)
    print("类人脑独白引擎测试")
    print("=" * 60)
    
    engine = MonologueEngineTest()
    
    # 测试 1: 初始状态
    print("\n【测试 1】初始状态检查")
    print(f"  思维状态: {engine.current_thought_state.value}")
    print(f"  情绪状态: {engine.current_emotion.value}")
    print(f"  当前主题: {engine.current_theme.content if engine.current_theme else 'None'}")
    print(f"  当前概念: {engine.current_concept}")
    
    # 测试 2: 状态转换
    print("\n【测试 2】状态转换测试 (10次)")
    state_changes = []
    for i in range(10):
        old_state = engine.current_thought_state
        engine._transition_thought_state()
        new_state = engine.current_thought_state
        if old_state != new_state:
            state_changes.append(f"{old_state.value} → {new_state.value}")
    
    print(f"  状态变化次数: {len(state_changes)}")
    for change in state_changes[:5]:
        print(f"    {change}")
    
    # 测试 3: 外部刺激处理
    print("\n【测试 3】外部刺激处理")
    engine = MonologueEngineTest()  # 重置
    result = engine.simulate_monologue("什么是意识？")
    print(f"  输入: '什么是意识？'")
    print(f"  思维状态: {result['old_state']} → {result['new_state']}")
    print(f"  情绪状态: {result['emotion']}")
    print(f"  新主题: {result['theme']}")
    print(f"  生成的Prompt: {result['prompt']}")
    
    # 测试 4: 连续独白生成
    print("\n【测试 4】连续独白生成 (5轮)")
    engine = MonologueEngineTest()
    engine.simulate_monologue("思考一下时间的本质")
    
    for i in range(5):
        result = engine.simulate_monologue()
        print(f"\n  第 {i+1} 轮:")
        print(f"    状态: {result['new_state']}")
        print(f"    情绪: {result['emotion']}")
        print(f"    Prompt: {result['prompt'][:60]}...")
    
    # 测试 5: 主题漂移和回归
    print("\n【测试 5】主题漂移和回归测试")
    engine = MonologueEngineTest()
    engine._set_new_theme("人工智能的伦理", importance=0.9)
    
    # 强制漂移
    engine.current_thought_state = ThoughtState.WANDERING
    for i in range(5):
        engine.simulate_monologue()
    
    print(f"  主题: {engine.current_theme.content}")
    print(f"  漂移次数: {engine.current_theme.drift_count}")
    print(f"  是否需要回归: {engine._should_return_to_theme()}")
    
    # 测试 6: 联想链
    print("\n【测试 6】联想链测试")
    engine = MonologueEngineTest()
    engine.current_concept = "意识"
    
    print(f"  起始概念: {engine.current_concept}")
    for i in range(5):
        next_concept, link_type = engine._generate_association(engine.current_concept)
        print(f"  联想 {i+1}: {engine.current_concept} --[{link_type}]--> {next_concept}")
        engine.current_concept = next_concept
    
    # 测试 7: 情绪影响
    print("\n【测试 7】情绪影响测试")
    test_contexts = [
        "为什么会这样？",
        "我发现了！",
        "这让我困惑...",
        "让我思考一下",
        "原来如此！"
    ]
    
    for ctx in test_contexts:
        engine._update_emotion_state(ctx)
        style = engine.emotion_style_prompts[engine.current_emotion]
        print(f"  输入: '{ctx}' → 情绪: {engine.current_emotion.value} → 风格: {style}")
    
    # 测试 8: 统计信息
    print("\n【测试 8】统计信息")
    engine = MonologueEngineTest()
    for i in range(20):
        engine.simulate_monologue()
    
    state_counts = {}
    for seg in engine.monologue_segments:
        state = seg.thought_state.value
        state_counts[state] = state_counts.get(state, 0) + 1
    
    print(f"  总独白数: {len(engine.monologue_segments)}")
    print(f"  状态分布:")
    for state, count in state_counts.items():
        print(f"    {state}: {count} ({count/len(engine.monologue_segments)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
