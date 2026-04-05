"""
人类记忆增强模块 - Human Memory Enhancement Module

基于认知神经科学原理，实现类人记忆机制的六个核心组件:

1. EbbinghausForgettingCurve —— 艾宾浩斯遗忘曲线
   - 科学遗忘模型: S = exp(-t / (strength * stability))
   - 初期快速遗忘 → 长期缓慢衰减
   - 支持间隔效应 (spacing effect) 复述强化

2. EmotionalMemoryModulator —— 情绪记忆调节器 (杏仁核-海马体交互)
   - 高唤醒情绪 (恐惧/快乐/愤怒) 产生更强记忆
   - 情绪强度随时间衰减，留下"情绪阴影"
   - 丰富的中文情绪关键词检测

3. ContextDependentMemory —— 语境依赖记忆 (状态依存性记忆)
   - 存储语境签名 (情绪/主题/时间段)
   - 基于语境相似度提升召回分数
   - 模拟人类"在相似环境下更易回忆"的特性

4. SpacingEffectManager —— 间隔效应管理器
   - 跟踪复述历史，优化复习间隔
   - 最优间隔序列: 1min → 10min → 1hr → 1day → 3day → 7day → 21day → 60day
   - 判断是否需要复述及推荐复述时机

5. MemoryInterferenceEngine —— 记忆干扰引擎
   - 前摄干扰: 旧记忆干扰新相似记忆
   - 倒摄干扰: 新记忆干扰旧相似记忆
   - 相似度越高干扰越大

6. SourceMonitor —— 记忆来源监控器
   - 追踪记忆来源: 用户告知 / AI生成 / 自身推理 / 外部知识
   - 来源置信度随时间衰减
   - 模拟人类"我知道你告诉过我"的来源归因能力

所有类均可独立导入和使用，与现有 HippocampusSystem 兼容。
"""

import time
import math
import re
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 1. 艾宾浩斯遗忘曲线 - Ebbinghaus Forgetting Curve
# ============================================================================

class EbbinghausForgettingCurve:
    """
    艾宾浩斯遗忘曲线实现

    科学背景:
    - 艾宾浩斯 (1885) 通过无意义音节实验发现记忆随时间指数衰减
    - 遗忘速度: 最初最快（1小时内遗忘约50%），之后逐渐放缓
    - 复述（间隔效应）可显著减缓遗忘速度，每次成功复述都会增加记忆稳定性

    核心公式:
        S = exp(-t / (strength * stability))

    其中:
        S = 记忆保持强度 (0.0 ~ 1.0)
        t = 距上次复述经过的时间 (秒)
        strength = 记忆强度因子（受情绪显著性和复述次数影响）
        stability = 记忆稳定度（每次复述后增加，模拟间隔效应）

    特性:
        - 初期陡峭: 第一次复述前遗忘速度极快（模拟人类1小时内遗忘50%的现象）
        - 长期平坦: 经过多次复述后，记忆趋于长期稳定
        - 间隔效应: 每次复述不仅重置时间，还增加 stability 参数
    """

    # 默认参数（基于艾宾浩斯实验数据的拟合参数）
    DEFAULT_INITIAL_STRENGTH: float = 0.5      # 初始记忆强度
    DEFAULT_INITIAL_STABILITY: float = 60.0     # 初始稳定度（秒）—— 1分钟内遗忘约63%
    STABILITY_GROWTH_FACTOR: float = 2.5        # 每次复述稳定度增长倍数
    STABILITY_GROWTH_CAP: float = 86400.0 * 365  # 稳定度上限（约1年）
    MINIMUM_RETENTION: float = 0.01             # 最低保留值（防止完全遗忘）

    def __init__(
        self,
        initial_strength: float = DEFAULT_INITIAL_STRENGTH,
        initial_stability: float = DEFAULT_INITIAL_STABILITY,
        emotional_salience: float = 0.0,
    ):
        """
        初始化遗忘曲线

        Args:
            initial_strength: 初始记忆强度 (0.1 ~ 2.0)
                - 值越大代表记忆越强烈
                - 受情绪显著性和初始编码深度影响
            initial_stability: 初始记忆稳定度 (秒)
                - 值越大代表记忆越持久
                - 初始值较小（1分钟），代表人类快速遗忘的初始阶段
            emotional_salience: 情绪显著性 (0.0 ~ 1.0)
                - 高情绪事件会同时增强 strength 和 stability
        """
        self.strength = initial_strength
        self.stability = initial_stability
        self.emotional_salience = emotional_salience

        # 应用情绪加成: 情绪显著事件产生更强、更稳定的记忆
        if emotional_salience > 0.5:
            self.strength *= (1.0 + emotional_salience * 0.8)
            self.stability *= (1.0 + emotional_salience * 0.5)

        self.rehearsal_count: int = 0
        self.last_rehearsal_time: Optional[float] = None  # 上次复述时间（Unix时间戳）

        # 历史记录（用于可视化/调试）
        self._rehearsal_history: List[float] = []  # 每次复述的时间戳

    def get_retention(
        self,
        timestamp: Optional[float] = None,
        last_rehearsal_time: Optional[float] = None,
    ) -> float:
        """
        计算当前记忆保持强度

        公式: S = exp(-t / (strength * stability))

        特点:
            - t=0 时 S=1.0（刚学习完保持100%）
            - t=strength*stability 时 S≈0.37（保留约37%）
            - 随着复述增加 stability，曲线变得越平坦

        Args:
            timestamp: 当前时间戳（秒），默认使用系统当前时间
            last_rehearsal_time: 上次复述时间戳，默认使用内部记录的 last_rehearsal_time

        Returns:
            S: 记忆保持强度 (0.0 ~ 1.0)
        """
        if timestamp is None:
            timestamp = time.time()

        if last_rehearsal_time is None:
            last_rehearsal_time = self.last_rehearsal_time

        if last_rehearsal_time is None:
            # 尚未复述过，使用当前时间作为起点
            return 1.0

        t = max(0.0, timestamp - last_rehearsal_time)

        # 核心遗忘公式
        denominator = self.strength * self.stability
        if denominator <= 0:
            return self.MINIMUM_RETENTION

        retention = math.exp(-t / denominator)

        # 确保不低于最低值（完全遗忘需要极长时间）
        retention = max(retention, self.MINIMUM_RETENTION)

        return retention

    def rehearse(
        self,
        timestamp: Optional[float] = None,
        success: bool = True,
    ) -> float:
        """
        执行一次复述（间隔效应实现）

        间隔效应原理:
            - 分散练习（间隔复习）比集中练习（突击复习）效果更好
            - 每次成功复述都会增加记忆的 stability（稳定度）
            - stability 增长因子随复述次数递减（边际效应递减）

        复述效果:
            - 成功复述: stability *= growth_factor (但增长因子随次数递减)
            - 失败复述: stability *= 0.5（遗忘加速，需要重新学习）

        Args:
            timestamp: 复述时间戳，默认使用系统当前时间
            success: 复述是否成功（能否正确回忆）

        Returns:
            new_stability: 更新后的稳定度
        """
        if timestamp is None:
            timestamp = time.time()

        self.rehearsal_count += 1
        self.last_rehearsal_time = timestamp
        self._rehearsal_history.append(timestamp)

        if success:
            # 间隔效应: 每次复述增加稳定度，但增量递减
            # 使用对数衰减的增长因子，模拟边际效应递减
            decay_factor = 1.0 / (1.0 + 0.15 * self.rehearsal_count)
            growth = self.STABILITY_GROWTH_FACTOR * decay_factor
            self.stability *= growth

            # 同时适度增强 strength（但幅度较小）
            self.strength = min(self.strength * 1.05, 3.0)
        else:
            # 复述失败: 稳定度和强度都下降
            self.stability = max(self.stability * 0.5, self.DEFAULT_INITIAL_STABILITY)
            self.strength = max(self.strength * 0.7, self.DEFAULT_INITIAL_STRENGTH * 0.5)

        # 限制稳定度上限（防止数值无限增长）
        self.stability = min(self.stability, self.STABILITY_GROWTH_CAP)

        return self.stability

    def get_time_since_last_rehearsal(self, timestamp: Optional[float] = None) -> float:
        """
        获取距上次复述经过的时间（秒）

        Args:
            timestamp: 当前时间戳

        Returns:
            经过的时间（秒），若从未复述返回 -1
        """
        if self.last_rehearsal_time is None:
            return -1.0

        if timestamp is None:
            timestamp = time.time()

        return max(0.0, timestamp - self.last_rehearsal_time)

    def get_curve_points(
        self,
        num_points: int = 100,
        time_span_hours: float = 24.0,
    ) -> List[Tuple[float, float]]:
        """
        生成遗忘曲线数据点（用于可视化）

        Args:
            num_points: 数据点数量
            time_span_hours: 时间跨度（小时）

        Returns:
            [(time_hours, retention), ...] 时间-保留率数据点列表
        """
        base_time = self.last_rehearsal_time or time.time()
        max_time = time_span_hours * 3600  # 转换为秒

        points = []
        for i in range(num_points):
            t = (i / num_points) * max_time
            s = self.get_retention(timestamp=base_time + t)
            points.append((t / 3600, s))  # 转换为小时

        return points

    def get_state(self) -> dict:
        """获取遗忘曲线状态（用于序列化/持久化）"""
        return {
            'strength': self.strength,
            'stability': self.stability,
            'emotional_salience': self.emotional_salience,
            'rehearsal_count': self.rehearsal_count,
            'last_rehearsal_time': self.last_rehearsal_time,
            'rehearsal_history': self._rehearsal_history,
        }

    def set_state(self, state: dict) -> None:
        """从状态字典恢复遗忘曲线"""
        self.strength = state.get('strength', self.DEFAULT_INITIAL_STRENGTH)
        self.stability = state.get('stability', self.DEFAULT_INITIAL_STABILITY)
        self.emotional_salience = state.get('emotional_salience', 0.0)
        self.rehearsal_count = state.get('rehearsal_count', 0)
        self.last_rehearsal_time = state.get('last_rehearsal_time', None)
        self._rehearsal_history = state.get('rehearsal_history', [])


# ============================================================================
# 2. 情绪记忆调节器 - Emotional Memory Modulator
# ============================================================================

class EmotionType(Enum):
    """情绪类型枚举"""
    JOY = "joy"               # 快乐
    FEAR = "fear"             # 恐惧
    SADNESS = "sadness"       # 悲伤
    ANGER = "anger"           # 愤怒
    SURPRISE = "surprise"     # 惊讶
    DISGUST = "disgust"       # 厌恶
    NEUTRAL = "neutral"       # 中性


class EmotionalMemoryModulator:
    """
    情绪记忆调节器 (杏仁核-海马体交互模型)

    科学背景:
        - 杏仁核 (amygdala) 在情绪体验时释放肾上腺素和皮质醇
        - 这些激素增强海马体 (hippocampus) 的记忆编码和巩固
        - 高唤醒 (high-arousal) 情绪（恐惧、快乐、愤怒）产生最强的记忆增强
        - 低唤醒 (low-arousal) 情绪（悲伤、厌恶）增强效应较弱
        - 情绪强度随时间衰减，但会留下"情绪阴影" (emotional shadow)
          即情绪的语义内容被记住，但感受强度下降

    核心功能:
        - 根据情绪类型和强度调节记忆强度
        - 模拟情绪的时间衰减（情绪阴影效应）
        - 支持中英文情绪关键词检测
    """

    # 情绪的唤醒度 (arousal) 映射: 值越高 → 记忆增强越强
    # 基于 Russell 情绪环形模型
    EMOTION_AROUSAL: Dict[str, float] = {
        'joy': 0.85,        # 高唤醒
        'fear': 0.95,       # 最高唤醒（进化上最重要的记忆）
        'sadness': 0.30,    # 低唤醒
        'anger': 0.80,      # 高唤醒
        'surprise': 0.90,   # 高唤醒
        'disgust': 0.40,    # 中低唤醒
        'neutral': 0.0,     # 无唤醒
    }

    # 情绪衰减半衰期（秒）—— 不同情绪的衰减速度不同
    EMOTION_DECAY_HALFLIFE: Dict[str, float] = {
        'joy': 86400.0 * 3,          # 快乐: 约3天
        'fear': 86400.0 * 30,        # 恐惧: 约30天（恐惧记忆最持久）
        'sadness': 86400.0 * 7,      # 悲伤: 约7天
        'anger': 86400.0 * 2,        # 愤怒: 约2天
        'surprise': 86400.0 * 1,     # 惊讶: 约1天（最短）
        'disgust': 86400.0 * 5,      # 厌恶: 约5天
        'neutral': 3600.0,           # 中性: 约1小时
    }

    # 情绪阴影基准值（情绪衰减后留下的永久增强）
    EMOTIONAL_SHADOW: Dict[str, float] = {
        'joy': 0.15,
        'fear': 0.30,      # 恐惧留下的阴影最深
        'sadness': 0.10,
        'anger': 0.20,
        'surprise': 0.10,
        'disgust': 0.10,
        'neutral': 0.0,
    }

    # ===== 中文情绪关键词库 =====
    CHINESE_EMOTION_KEYWORDS: Dict[str, List[str]] = {
        'joy': [
            '开心', '高兴', '快乐', '幸福', '欣喜', '兴奋', '愉快', '欢乐',
            '满足', '欣慰', '喜悦', '爽', '棒', '太好了', '哈哈', '嘻嘻',
            '乐', '美好', '赞', '不错', '喜欢', '爱', '美好', '感恩',
            '庆祝', '恭喜', '好运', '幸运', '甜', '笑', '乐趣', '享受',
            '欣慰', '舒适', '自在', '惬意', '愉悦', '畅快', '振奋', '欢喜',
        ],
        'fear': [
            '害怕', '恐惧', '担心', '焦虑', '紧张', '恐怖', '吓', '怕',
            '不安', '惊恐', '畏惧', '慌', '危险', '可怕', '吓人', '噩梦',
            '慌张', '忐忑', '心慌', '发抖', '战栗', '可怕', '毛骨悚然',
            '心惊肉跳', '提心吊胆', '不寒而栗', '惶恐', '忧心', '惧怕',
        ],
        'sadness': [
            '难过', '悲伤', '伤心', '哭', '痛苦', '失望', '沮丧', '忧伤',
            '心碎', '忧郁', '可怜', '惋惜', '遗憾', '郁闷', '低落', '惆怅',
            '痛心', '哀伤', '落泪', '心酸', '凄凉', '悲哀', '苦', '惨',
            '辛酸', '悲', '哀', '凄惨', '悲痛', '消沉', '落寞', '孤寂',
        ],
        'anger': [
            '生气', '愤怒', '恼火', '烦', '气死', '讨厌', '恨', '暴怒',
            '不满', '气愤', '发火', '火大', '怒', '可恶', '该死', '混蛋',
            '烦躁', '暴躁', '愤恨', '恼怒', '怒火', '气炸', '忍无可忍',
            '怒不可遏', '气急败坏', '愤怒', '愤慨', '不爽', '窝火',
        ],
        'surprise': [
            '惊讶', '震惊', '没想到', '意外', '吃惊', '天哪', '不会吧',
            '竟然', '居然', '不可思议', '难以置信', '哇', '天啊', '我的天',
            '出乎意料', '万万没想到', '大吃一惊', '目瞪口呆', '惊讶',
            '吃惊', '没想到', '惊喜', '诧异', '愕然', '震惊',
        ],
        'disgust': [
            '恶心', '厌恶', '反感', '鄙视', '瞧不起', '吐', '受不了',
            '别扭', '膈应', '作呕', '嫌弃', '令人作呕', '讨厌', '龌龊',
            '下流', '卑鄙', '肮脏', '污秽', '不堪', '排斥', '抗拒',
        ],
    }

    # 英文情绪关键词（辅助）
    ENGLISH_EMOTION_KEYWORDS: Dict[str, List[str]] = {
        'joy': ['happy', 'joy', 'love', 'great', 'wonderful', 'amazing', 'excited',
                'fantastic', 'awesome', 'delighted', 'pleased', 'cheerful', 'glad'],
        'fear': ['afraid', 'scared', 'fear', 'terrified', 'anxious', 'worried',
                 'panic', 'horror', 'dread', 'frightened', 'nervous'],
        'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'heartbroken',
                    'grief', 'sorrow', 'lonely', 'disappointed', 'hopeless'],
        'anger': ['angry', 'furious', 'mad', 'annoyed', 'irritated', 'outraged',
                  'hate', 'rage', 'frustrated', 'enraged'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected',
                     'wow', 'incredible', 'unbelievable'],
        'disgust': ['disgusted', 'gross', 'nasty', 'revolting', 'repulsive',
                    'sick', 'eww', 'horrible'],
    }

    def __init__(
        self,
        base_emotion_decay_rate: float = 0.0001,
        shadow_decay_enabled: bool = True,
    ):
        """
        初始化情绪记忆调节器

        Args:
            base_emotion_decay_rate: 基础情绪衰减速率
            shadow_decay_enabled: 是否启用情绪阴影效应
        """
        self.base_decay_rate = base_emotion_decay_rate
        self.shadow_decay_enabled = shadow_decay_enabled

        # 情绪关键词正则缓存（优化性能）
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """预编译情绪关键词正则表达式（提升匹配性能）"""
        all_keywords = {}
        for emotion_type, keywords in self.CHINESE_EMOTION_KEYWORDS.items():
            all_keywords.setdefault(emotion_type, []).extend(keywords)
        for emotion_type, keywords in self.ENGLISH_EMOTION_KEYWORDS.items():
            all_keywords.setdefault(emotion_type, []).extend(keywords)

        for emotion_type, keywords in all_keywords.items():
            patterns = []
            for kw in keywords:
                try:
                    patterns.append(re.compile(re.escape(kw), re.IGNORECASE))
                except re.error:
                    pass
            self._compiled_patterns[emotion_type] = patterns

    def detect_emotion(self, text: str) -> Tuple[str, float]:
        """
        从文本中检测情绪类型和强度

        Args:
            text: 输入文本（支持中英文混合）

        Returns:
            (emotion_type, intensity): 情绪类型和强度 (0.0 ~ 1.0)
        """
        if not text:
            return ('neutral', 0.0)

        emotion_scores: Dict[str, float] = {}
        total_matches = 0

        for emotion_type, patterns in self._compiled_patterns.items():
            count = 0
            for pattern in patterns:
                matches = pattern.findall(text)
                count += len(matches)

            if count > 0:
                emotion_scores[emotion_type] = count
                total_matches += count

        if not emotion_scores:
            return ('neutral', 0.0)

        # 归一化为强度值 (0.0 ~ 1.0)
        # 多个关键词匹配 → 强度更高（但有上限）
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        raw_intensity = emotion_scores[best_emotion] / max(total_matches, 1)

        # 使用对数缩放将原始计数映射到 [0.1, 1.0] 区间
        match_count = emotion_scores[best_emotion]
        intensity = min(0.1 + 0.15 * math.log1p(match_count * 3), 1.0)

        # 考虑唤醒度修正: 高唤醒情绪的检测阈值更低
        arousal = self.EMOTION_AROUSAL.get(best_emotion, 0.5)
        intensity = intensity * (0.5 + 0.5 * arousal)

        return (best_emotion, round(min(intensity, 1.0), 3))

    def modulate_strength(
        self,
        base_strength: float,
        emotion_type: str,
        intensity: float,
        time_since_event: float,
    ) -> float:
        """
        根据情绪调节记忆强度

        模拟杏仁核-海马体交互:
            1. 获取情绪唤醒度 (arousal)
            2. 计算情绪随时间的衰减（指数衰减 + 情绪阴影）
            3. 将衰减后的情绪影响叠加到基础记忆强度上

        Args:
            base_strength: 基础记忆强度
            emotion_type: 情绪类型 (joy/fear/sadness/anger/surprise/disgust/neutral)
            intensity: 原始情绪强度 (0.0 ~ 1.0)
            time_since_event: 事件发生以来的时间（秒）

        Returns:
            modulated_strength: 调节后的记忆强度
        """
        if emotion_type == 'neutral' or intensity <= 0.0:
            return base_strength

        # 1. 获取情绪唤醒度
        arousal = self.EMOTION_AROUSAL.get(emotion_type, 0.0)

        # 2. 计算情绪衰减
        #    使用指数衰减: I(t) = I_0 * exp(-λt)
        #    λ = ln(2) / half_life
        half_life = self.EMOTION_DECAY_HALFLIFE.get(emotion_type, 86400.0)
        decay_constant = math.log(2) / max(half_life, 1.0)
        decayed_intensity = intensity * math.exp(-decay_constant * time_since_event)

        # 3. 计算情绪阴影（永久保留的情绪增强）
        emotional_shadow = 0.0
        if self.shadow_decay_enabled:
            shadow = self.EMOTIONAL_SHADOW.get(emotion_type, 0.0)
            # 情绪阴影也随时间衰减，但速度极慢
            shadow_decay = math.exp(-self.base_decay_rate * time_since_event)
            emotional_shadow = shadow * intensity * shadow_decay

        # 4. 计算情绪对记忆的增强系数
        #    增强幅度 = (衰减后的情绪强度 * 唤醒度) + 情绪阴影
        emotion_boost = (
            decayed_intensity * arousal * intensity
            + emotional_shadow
        )

        # 5. 应用调节: 基础强度 + 情绪增强
        modulated = base_strength * (1.0 + emotion_boost)

        return modulated

    def get_emotional_summary(self, text: str) -> Dict[str, Any]:
        """
        获取文本的完整情绪分析报告

        Args:
            text: 输入文本

        Returns:
            情绪分析报告字典，包含:
            - dominant_emotion: 主导情绪
            - intensity: 情绪强度
            - arousal: 唤醒度
            - all_emotions: 所有检测到的情绪及其分数
        """
        emotion_type, intensity = self.detect_emotion(text)
        arousal = self.EMOTION_AROUSAL.get(emotion_type, 0.0)

        # 检测所有情绪（不仅是最强的）
        all_emotions: Dict[str, float] = {}
        for etype, patterns in self._compiled_patterns.items():
            count = sum(len(p.findall(text)) for p in patterns)
            if count > 0:
                all_emotions[etype] = min(0.1 + 0.15 * math.log1p(count * 3), 1.0)

        return {
            'dominant_emotion': emotion_type,
            'intensity': intensity,
            'arousal': arousal,
            'all_emotions': all_emotions,
            'is_high_arousal': arousal >= 0.7,
        }


# ============================================================================
# 3. 语境依赖记忆 - Context Dependent Memory
# ============================================================================

class ContextDependentMemory:
    """
    语境依赖记忆 (状态依存性记忆 / Context-Dependent Memory)

    科学背景:
        - Godden & Baddeley (1975): 潜水员在水下学习的单词，在水下回忆效果最好
        - 人类在相同的心境、环境、时间下更容易回忆起相关信息
        - 状态依存性记忆 (State-Dependent Memory): 内部状态（情绪、生理）影响回忆
        - 环境依存性记忆 (Context-Dependent Memory): 外部环境影响回忆

    核心机制:
        1. 存储时记录"语境签名" (context signature): 情绪 + 主题 + 时间段
        2. 召回时计算当前语境与存储语境的相似度
        3. 相似语境下的记忆获得额外召回加成

    语境签名的维度:
        - mood: 当前情绪状态 (0.0 ~ 1.0 向量)
        - topic: 话题/主题 (文本)
        - time_of_day: 一天中的时段 (0-23)
        - day_of_week: 星期几 (0-6)
        - setting: 场景/环境 (文本)
    """

    # 时间段划分
    TIME_PERIODS = {
        'early_morning': (5, 8),     # 清晨 5:00-8:00
        'morning': (8, 12),          # 上午 8:00-12:00
        'afternoon': (12, 17),       # 下午 12:00-17:00
        'evening': (17, 21),         # 傍晚 17:00-21:00
        'night': (21, 24),           # 夜晚 21:00-24:00
        'late_night': (0, 5),        # 深夜 0:00-5:00
    }

    def __init__(
        self,
        mood_weight: float = 0.35,
        topic_weight: float = 0.35,
        time_weight: float = 0.20,
        setting_weight: float = 0.10,
    ):
        """
        初始化语境依赖记忆

        Args:
            mood_weight: 情绪维度权重
            topic_weight: 主题维度权重
            time_weight: 时间维度权重
            setting_weight: 场景维度权重
        """
        self.mood_weight = mood_weight
        self.topic_weight = topic_weight
        self.time_weight = time_weight
        self.setting_weight = setting_weight

        # 情绪-语境关联映射（用于快速情绪比较）
        self._emotion_mood_map = {
            'joy': {'valence': 0.9, 'arousal': 0.7},
            'fear': {'valence': -0.7, 'arousal': 0.9},
            'sadness': {'valence': -0.8, 'arousal': 0.2},
            'anger': {'valence': -0.6, 'arousal': 0.8},
            'surprise': {'valence': 0.1, 'arousal': 0.9},
            'disgust': {'valence': -0.7, 'arousal': 0.4},
            'neutral': {'valence': 0.0, 'arousal': 0.1},
        }

        # 主题关键词（用于快速比较主题相似度）
        self._topic_categories: Dict[str, List[str]] = {
            'work': ['工作', '上班', '项目', '会议', '同事', '老板', '加班', 'job', 'work', 'meeting'],
            'life': ['生活', '日常', '家务', '做饭', '购物', 'life', 'daily', 'cook'],
            'relationship': ['感情', '恋爱', '朋友', '家人', 'love', 'friend', 'family', 'relationship'],
            'health': ['健康', '运动', '锻炼', '生病', 'health', 'exercise', 'sick'],
            'study': ['学习', '读书', '考试', '课程', 'study', 'learn', 'exam', 'course'],
            'entertainment': ['游戏', '电影', '音乐', '娱乐', 'game', 'movie', 'music'],
            'technology': ['编程', '代码', 'AI', '技术', 'programming', 'code', 'tech'],
            'travel': ['旅行', '旅游', '出行', 'trip', 'travel', 'vacation'],
            'food': ['美食', '餐厅', '吃', 'food', 'restaurant', 'eat'],
            'finance': ['投资', '理财', '股票', '钱', 'investment', 'money', 'stock'],
        }

    def extract_context_signature(
        self,
        text: str,
        emotion_tag: str = "neutral",
        timestamp: Optional[float] = None,
        setting: str = "",
    ) -> Dict[str, Any]:
        """
        从文本和环境信息中提取语境签名

        Args:
            text: 输入文本
            emotion_tag: 情绪标签 (joy/fear/sadness/anger/surprise/disgust/neutral)
            timestamp: 时间戳，默认使用当前时间
            setting: 场景描述（可选）

        Returns:
            context_signature: 语境签名字典
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. 情绪维度 (valence-arousal 模型)
        mood_info = self._emotion_mood_map.get(emotion_tag, {'valence': 0.0, 'arousal': 0.1})
        mood_vector = np.array([mood_info['valence'], mood_info['arousal']], dtype=np.float32)

        # 2. 主题维度 (基于关键词匹配的多分类向量)
        topic_vector = self._compute_topic_vector(text)

        # 3. 时间维度
        hour = int(time.localtime(timestamp).tm_hour)
        time_vector = self._compute_time_vector(hour)

        # 4. 识别主要主题（用于后续比较）
        primary_topic = self._detect_primary_topic(text)

        return {
            'mood_vector': mood_vector.tolist(),
            'topic_vector': topic_vector.tolist(),
            'time_vector': time_vector.tolist(),
            'primary_topic': primary_topic,
            'hour': hour,
            'day_of_week': int(time.localtime(timestamp).tm_wday),
            'emotion_tag': emotion_tag,
            'setting': setting,
            'timestamp': timestamp,
        }

    def compute_context_boost(
        self,
        stored_context: Dict[str, Any],
        current_context: Dict[str, Any],
    ) -> float:
        """
        计算语境加成系数

        基于存储语境和当前语境的相似度，计算召回加成。
        相似度越高，加成越大（0.0 ~ 0.5）。

        计算方式:
            1. 情绪相似度: valence-arousal 向量的余弦相似度
            2. 主题相似度: 主题向量的余弦相似度
            3. 时间相似度: 时间段的重叠程度
            4. 综合加权求和

        Args:
            stored_context: 存储的语境签名
            current_context: 当前的语境签名

        Returns:
            boost: 召回加成系数 (0.0 ~ 0.5)
        """
        if not stored_context or not current_context:
            return 0.0

        # 1. 情绪相似度
        mood_sim = self._compute_mood_similarity(
            stored_context.get('mood_vector', [0, 0]),
            current_context.get('mood_vector', [0, 0]),
        )

        # 2. 主题相似度
        topic_sim = self._compute_topic_similarity(
            stored_context.get('topic_vector', [0] * 10),
            current_context.get('topic_vector', [0] * 10),
        )

        # 3. 时间相似度
        time_sim = self._compute_time_similarity(
            stored_context.get('time_vector', [0] * 6),
            current_context.get('time_vector', [0] * 6),
        )

        # 4. 主题匹配加成（完全匹配额外加成）
        topic_match_bonus = 0.0
        stored_topic = stored_context.get('primary_topic', '')
        current_topic = current_context.get('primary_topic', '')
        if stored_topic and current_topic and stored_topic == current_topic:
            topic_match_bonus = 0.1

        # 5. 综合加权
        total_similarity = (
            self.mood_weight * mood_sim
            + self.topic_weight * topic_sim
            + self.time_weight * time_sim
        )

        # 6. 映射到加成系数 (0.0 ~ 0.5)
        boost = total_similarity * 0.35 + topic_match_bonus
        boost = max(0.0, min(0.5, boost))

        return boost

    def _compute_topic_vector(self, text: str) -> np.ndarray:
        """计算文本的主题向量（10维，对应10个主题类别）"""
        vector = np.zeros(len(self._topic_categories), dtype=np.float32)
        text_lower = text.lower()

        for i, (category, keywords) in enumerate(self._topic_categories.items()):
            count = sum(1 for kw in keywords if kw.lower() in text_lower)
            if count > 0:
                vector[i] = min(count * 0.3, 1.0)

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _compute_time_vector(self, hour: int) -> np.ndarray:
        """计算时间段向量（6维，对应6个时段）"""
        vector = np.zeros(len(self.TIME_PERIODS), dtype=np.float32)

        for i, (period, (start, end)) in enumerate(self.TIME_PERIODS.items()):
            if start <= hour < end:
                vector[i] = 1.0
            # 相邻时段有微弱的交叉激活
            elif period != 'late_night' and start - 1 <= hour < end + 1:
                vector[i] = 0.3

        return vector

    def _compute_mood_similarity(
        self,
        mood1: List[float],
        mood2: List[float],
    ) -> float:
        """计算情绪向量相似度（余弦相似度）"""
        v1 = np.array(mood1, dtype=np.float32)
        v2 = np.array(mood2, dtype=np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        sim = float(np.dot(v1, v2) / (norm1 * norm2))
        # 将 [-1, 1] 映射到 [0, 1]
        return max(0.0, (sim + 1.0) / 2.0)

    def _compute_topic_similarity(
        self,
        topic1: List[float],
        topic2: List[float],
    ) -> float:
        """计算主题向量相似度（余弦相似度）"""
        v1 = np.array(topic1, dtype=np.float32)
        v2 = np.array(topic2, dtype=np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _compute_time_similarity(
        self,
        time1: List[float],
        time2: List[float],
    ) -> float:
        """计算时间向量相似度（重叠程度）"""
        v1 = np.array(time1, dtype=np.float32)
        v2 = np.array(time2, dtype=np.float32)

        # 使用内积作为重叠度量
        return float(np.dot(v1, v2))

    def _detect_primary_topic(self, text: str) -> str:
        """检测文本的主要主题"""
        text_lower = text.lower()

        best_topic = ""
        best_score = 0

        for category, keywords in self._topic_categories.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_topic = category

        return best_topic


# ============================================================================
# 4. 间隔效应管理器 - Spacing Effect Manager
# ============================================================================

@dataclass
class RehearsalRecord:
    """复述记录数据结构"""
    memory_id: str
    rehearsal_times: List[float] = field(default_factory=list)  # 每次复述的时间戳
    rehearsal_count: int = 0
    created_time: Optional[float] = None
    next_rehearsal_time: Optional[float] = None  # 下次推荐复述时间


class SpacingEffectManager:
    """
    间隔效应管理器 (Spaced Repetition / 分布式练习)

    科学背景:
        - Ebbinghaus 遗忘实验发现，分散复习比集中复习效果好 2-3 倍
        - SuperMemo (SM-2) 算法: 基于记忆表现动态调整复习间隔
        - Anki 间隔重复系统广泛用于语言学习和医学教育
        - 最优间隔序列: 随着复习次数增加，间隔指数增长

    最优复习间隔序列 (基于认知心理学研究):
        1min → 10min → 1hr → 1day → 3day → 7day → 21day → 60day

    核心功能:
        - 跟踪每条记忆的复述历史
        - 判断记忆是否需要复述
        - 推荐最优复述时间
        - 记录复述事件
    """

    # 最优间隔序列（秒）—— 基于认知心理学研究的拟合数据
    OPTIMAL_INTERVALS: List[float] = [
        60,                   # 1 分钟
        600,                  # 10 分钟
        3600,                 # 1 小时
        86400,                # 1 天
        86400 * 3,            # 3 天
        86400 * 7,            # 7 天
        86400 * 21,           # 21 天
        86400 * 60,           # 60 天
        86400 * 180,          # 180 天（半年，极限间隔）
    ]

    # 间隔容忍度（在推荐时间前后多少比例内算"及时"复述）
    INTERVAL_TOLERANCE: float = 0.3  # 30%

    def __init__(self, max_memories: int = 50000):
        """
        初始化间隔效应管理器

        Args:
            max_memories: 最大管理的记忆数量
        """
        self.max_memories = max_memories
        self._records: Dict[str, RehearsalRecord] = {}

    def get_optimal_rehearsal_interval(self, rehearsal_count: int) -> float:
        """
        获取第 N 次复述的最优间隔（秒）

        使用指数增长的间隔序列，模拟认知心理学中的最优复习间隔。

        Args:
            rehearsal_count: 已完成的复述次数（从0开始）

        Returns:
            interval_seconds: 最优间隔时间（秒）
        """
        idx = min(rehearsal_count, len(self.OPTIMAL_INTERVALS) - 1)
        return self.OPTIMAL_INTERVALS[idx]

    def should_rehearse(
        self,
        memory_id: str,
        current_time: Optional[float] = None,
    ) -> bool:
        """
        判断记忆是否应该复述

        当当前时间超过推荐复述时间时返回 True。

        Args:
            memory_id: 记忆 ID
            current_time: 当前时间戳，默认使用系统时间

        Returns:
            should: 是否需要复述
        """
        if current_time is None:
            current_time = time.time()

        record = self._records.get(memory_id)
        if record is None:
            # 新记忆，尚未复述过，应在第一次间隔后复述
            return True

        if record.next_rehearsal_time is None:
            return True

        return current_time >= record.next_rehearsal_time

    def record_rehearsal(
        self,
        memory_id: str,
        timestamp: Optional[float] = None,
        success: bool = True,
    ) -> RehearsalRecord:
        """
        记录一次复述事件

        复述后根据结果更新下次推荐复述时间:
        - 成功复述: 间隔增加（进入下一个间隔级别）
        - 失败复述: 间隔重置（回到第一个间隔级别）

        Args:
            memory_id: 记忆 ID
            timestamp: 复述时间戳，默认使用系统时间
            success: 复述是否成功

        Returns:
            record: 更新后的复述记录
        """
        if timestamp is None:
            timestamp = time.time()

        record = self._records.get(memory_id)
        if record is None:
            record = RehearsalRecord(
                memory_id=memory_id,
                created_time=timestamp,
            )
            if len(self._records) >= self.max_memories:
                # 淘汰最旧的记录
                oldest_id = min(self._records, key=lambda k: self._records[k].created_time or 0)
                del self._records[oldest_id]
            self._records[memory_id] = record

        # 记录复述时间
        record.rehearsal_times.append(timestamp)
        record.rehearsal_count += 1

        if success:
            # 成功: 使用下一个间隔级别
            next_interval = self.get_optimal_rehearsal_interval(record.rehearsal_count)
            record.next_rehearsal_time = timestamp + next_interval
        else:
            # 失败: 重置到第一个间隔级别
            first_interval = self.get_optimal_rehearsal_interval(0)
            record.next_rehearsal_time = timestamp + first_interval

        return record

    def get_rehearsal_schedule(
        self,
        memory_id: str,
    ) -> List[float]:
        """
        获取记忆的完整复述时间表

        返回基于当前复述进度的未来所有推荐复述时间点。

        Args:
            memory_id: 记忆 ID

        Returns:
            schedule: 推荐复述时间戳列表
        """
        record = self._records.get(memory_id)
        if record is None:
            return []

        schedule = []
        current_count = record.rehearsal_count
        base_time = record.next_rehearsal_time or time.time()

        # 生成未来10个复述时间点
        for i in range(10):
            interval = self.get_optimal_rehearsal_interval(current_count + i)
            schedule.append(base_time + interval)
            base_time += interval

        return schedule

    def get_rehearsal_info(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        获取记忆的复述统计信息

        Args:
            memory_id: 记忆 ID

        Returns:
            info: 复述信息字典，记忆不存在时返回 None
        """
        record = self._records.get(memory_id)
        if record is None:
            return None

        next_interval = self.get_optimal_rehearsal_interval(record.rehearsal_count)
        current_interval = self.OPTIMAL_INTERVALS[
            min(record.rehearsal_count - 1, len(self.OPTIMAL_INTERVALS) - 1)
        ] if record.rehearsal_count > 0 else 0

        return {
            'memory_id': record.memory_id,
            'rehearsal_count': record.rehearsal_count,
            'created_time': record.created_time,
            'last_rehearsal_time': record.rehearsal_times[-1] if record.rehearsal_times else None,
            'current_interval': current_interval,
            'next_interval': next_interval,
            'next_rehearsal_time': record.next_rehearsal_time,
            'total_rehearsal_time': sum(
                record.rehearsal_times[i] - record.rehearsal_times[i - 1]
                for i in range(1, len(record.rehearsal_times))
            ) if len(record.rehearsal_times) > 1 else 0,
            'retention_level': min(record.rehearsal_count / 7.0, 1.0),  # 简化的记忆保持估计
        }

    def get_all_due_memories(
        self,
        current_time: Optional[float] = None,
    ) -> List[str]:
        """
        获取所有到期需要复述的记忆 ID 列表

        Args:
            current_time: 当前时间戳

        Returns:
            due_ids: 到期记忆 ID 列表（按紧急程度排序）
        """
        if current_time is None:
            current_time = time.time()

        due = []
        for memory_id, record in self._records.items():
            if self.should_rehearse(memory_id, current_time):
                overdue = 0.0
                if record.next_rehearsal_time:
                    overdue = current_time - record.next_rehearsal_time
                due.append((memory_id, overdue))

        # 按逾期程度排序（逾期越久越紧急）
        due.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in due]

    def remove_memory(self, memory_id: str) -> bool:
        """移除记忆的复述记录"""
        if memory_id in self._records:
            del self._records[memory_id]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取间隔效应管理器统计信息"""
        total_records = len(self._records)
        if total_records == 0:
            return {
                'total_memories': 0,
                'avg_rehearsal_count': 0.0,
                'due_count': 0,
                'mastery_distribution': {'new': 0, 'learning': 0, 'mature': 0, 'long_term': 0},
            }

        rehearsal_counts = [r.rehearsal_count for r in self._records.values()]
        avg_rehearsals = sum(rehearsal_counts) / total_records
        due_count = len(self.get_all_due_memories())

        # 记忆掌握程度分布
        mastery = {
            'new': sum(1 for c in rehearsal_counts if c == 0),
            'learning': sum(1 for c in rehearsal_counts if 1 <= c <= 3),
            'mature': sum(1 for c in rehearsal_counts if 4 <= c <= 6),
            'long_term': sum(1 for c in rehearsal_counts if c >= 7),
        }

        return {
            'total_memories': total_records,
            'avg_rehearsal_count': round(avg_rehearsals, 2),
            'due_count': due_count,
            'mastery_distribution': mastery,
        }

    def get_state(self) -> Dict[str, Any]:
        """获取管理器状态（用于序列化）"""
        records_state = {}
        for mid, record in self._records.items():
            records_state[mid] = {
                'memory_id': record.memory_id,
                'rehearsal_times': record.rehearsal_times,
                'rehearsal_count': record.rehearsal_count,
                'created_time': record.created_time,
                'next_rehearsal_time': record.next_rehearsal_time,
            }
        return {'records': records_state}

    def set_state(self, state: Dict[str, Any]) -> None:
        """从状态字典恢复管理器"""
        records_state = state.get('records', {})
        self._records.clear()
        for mid, rdata in records_state.items():
            record = RehearsalRecord(
                memory_id=rdata.get('memory_id', mid),
                rehearsal_times=rdata.get('rehearsal_times', []),
                rehearsal_count=rdata.get('rehearsal_count', 0),
                created_time=rdata.get('created_time', None),
                next_rehearsal_time=rdata.get('next_rehearsal_time', None),
            )
            self._records[mid] = record


# ============================================================================
# 5. 记忆干扰引擎 - Memory Interference Engine
# ============================================================================

class MemoryInterferenceEngine:
    """
    记忆干扰引擎 (前摄干扰 + 倒摄干扰)

    科学背景:
        - 前摄干扰 (Proactive Interference): 旧记忆阻碍新相似记忆的编码
          例: 学了法语后再学西班牙语，法语记忆干扰西班牙语学习
        - 倒摄干扰 (Retroactive Interference): 新记忆削弱旧相似记忆的回忆
          例: 学了新密码后，旧密码记忆变得模糊
        - 干扰强度与记忆间的相似度正相关
        - McGeoch (1942) 的干扰理论: 相似性是干扰的核心因素

    应用场景:
        - 当存储新记忆时，检查与已有记忆的相似度，标记可能的前摄干扰
        - 定期计算已有记忆间的倒摄干扰，降低被干扰记忆的激活强度
        - 帮助识别需要通过复述来巩固的"被威胁"记忆
    """

    # 干扰参数
    PROACTIVE_INTERFERENCE_FACTOR: float = 0.15   # 前摄干扰衰减因子
    RETROACTIVE_INTERFERENCE_FACTOR: float = 0.10  # 倒摄干扰衰减因子
    INTERFERENCE_DECAY_RATE: float = 0.995        # 干扰效果本身的衰减率（随时间减弱）
    SIMILARITY_THRESHOLD: float = 0.3             # 最低相似度阈值（低于此值不产生干扰）

    def __init__(
        self,
        proactive_factor: float = PROACTIVE_INTERFERENCE_FACTOR,
        retroactive_factor: float = RETROACTIVE_INTERFERENCE_FACTOR,
    ):
        """
        初始化记忆干扰引擎

        Args:
            proactive_factor: 前摄干扰强度因子
            retroactive_factor: 倒摄干扰强度因子
        """
        self.proactive_factor = proactive_factor
        self.retroactive_factor = retroactive_factor

        # 干扰记录: {(memory_id_1, memory_id_2): {'strength': float, 'timestamp': float, 'type': str}}
        self._interference_records: Dict[str, Dict[str, Any]] = {}

    def compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        计算两段文本之间的语义相似度

        使用多重信号组合:
        1. Jaccard 相似度 (字符级 n-gram)
        2. 关键词重叠度
        3. 长度相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            similarity: 相似度 (0.0 ~ 1.0)
        """
        if not text1 or not text2:
            return 0.0

        # 1. 字符级 Jaccard 相似度（bigram）
        def get_char_bigrams(text: str) -> Set[str]:
            return {text[i:i+2] for i in range(len(text) - 1)}

        bigrams1 = get_char_bigrams(text1.lower())
        bigrams2 = get_char_bigrams(text2.lower())

        if not bigrams1 or not bigrams2:
            return 0.0

        jaccard = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)

        # 2. 关键词重叠度（中文分词简易版）
        def tokenize(text: str) -> Set[str]:
            # 提取中文词和英文单词
            chinese = set(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
            english = set(re.findall(r'[a-zA-Z]{2,}', text.lower()))
            numbers = set(re.findall(r'\d+(?:\.\d+)?', text))
            return chinese | english | numbers

        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)

        if not tokens1 or not tokens2:
            keyword_overlap = 0.0
        else:
            keyword_overlap = len(tokens1 & tokens2) / len(tokens1 | tokens2)

        # 3. 综合相似度（加权平均）
        similarity = 0.5 * jaccard + 0.5 * keyword_overlap

        return max(0.0, min(1.0, similarity))

    def compute_embedding_similarity(
        self,
        emb1: Optional[torch.Tensor],
        emb2: Optional[torch.Tensor],
    ) -> float:
        """
        计算 embedding 向量间的余弦相似度

        Args:
            emb1: embedding 向量1
            emb2: embedding 向量2

        Returns:
            similarity: 相似度 (0.0 ~ 1.0)，任一为 None 返回 0.0
        """
        if emb1 is None or emb2 is None:
            return 0.0

        try:
            if emb1.dim() == 1:
                emb1 = emb1.unsqueeze(0)
            if emb2.dim() == 1:
                emb2 = emb2.unsqueeze(0)

            # 处理维度不一致
            min_dim = min(emb1.shape[-1], emb2.shape[-1])
            emb1 = emb1[..., :min_dim].float()
            emb2 = emb2[..., :min_dim].float()

            sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
            return max(0.0, min(1.0, sim))
        except Exception:
            return 0.0

    def compute_interference(
        self,
        new_memory: Any,
        existing_memories: List[Any],
    ) -> Dict[str, Dict[str, float]]:
        """
        计算新记忆与已有记忆之间的干扰

        对每条已有记忆计算:
        - 前摄干扰分数 (proactive): 旧记忆对新记忆的干扰
        - 倒摄干扰分数 (retroactive): 新记忆对旧记忆的干扰

        Args:
            new_memory: 新记忆对象，需具有 content / user_input / semantic_summary 属性
            existing_memories: 已有记忆对象列表

        Returns:
            interference_scores: {memory_id: {'proactive': float, 'retroactive': float, 'similarity': float}}
        """
        # 提取新记忆的文本
        new_text = self._extract_memory_text(new_memory)
        new_emb = getattr(new_memory, 'semantic_embedding', None)

        if not new_text:
            return {}

        results: Dict[str, Dict[str, float]] = {}

        for existing_mem in existing_memories:
            mem_id = getattr(existing_mem, 'memory_id', '')
            if not mem_id:
                continue

            # 计算文本相似度
            existing_text = self._extract_memory_text(existing_mem)
            text_sim = self.compute_similarity(new_text, existing_text) if existing_text else 0.0

            # 计算 embedding 相似度（如果可用）
            existing_emb = getattr(existing_mem, 'semantic_embedding', None)
            emb_sim = self.compute_embedding_similarity(new_emb, existing_emb)

            # 取两者中的最大值作为综合相似度
            similarity = max(text_sim, emb_sim)

            if similarity < self.SIMILARITY_THRESHOLD:
                continue  # 低于阈值，不产生干扰

            # 前摄干扰: 旧记忆对新记忆的阻碍
            # 旧记忆越强（activation_strength 越高），干扰越大
            old_strength = getattr(existing_mem, 'activation_strength', 1.0)
            proactive = similarity * self.proactive_factor * min(old_strength, 2.0)

            # 倒摄干扰: 新记忆对旧记忆的削弱
            # 新记忆的创建时间越近，干扰越强
            new_strength = getattr(new_memory, 'activation_strength', 1.0)
            retroactive = similarity * self.retroactive_factor * min(new_strength, 2.0)

            results[mem_id] = {
                'proactive': round(proactive, 4),
                'retroactive': round(retroactive, 4),
                'similarity': round(similarity, 4),
            }

            # 记录干扰关系
            pair_key = self._make_pair_key(mem_id, getattr(new_memory, 'memory_id', 'new'))
            self._interference_records[pair_key] = {
                'strength': similarity,
                'proactive': proactive,
                'retroactive': retroactive,
                'timestamp': time.time(),
                'type': 'bidirectional',
            }

        return results

    def apply_interference_decay(
        self,
        memories: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        对记忆应用干扰衰减

        遍历所有干扰记录，对被干扰的记忆降低激活强度。
        干扰效果本身也会随时间衰减（INTERFERENCE_DECAY_RATE）。

        Args:
            memories: 记忆字典 {memory_id: memory_object}

        Returns:
            decay_applied: {memory_id: 衰减量} 记录每条记忆被衰减的量
        """
        current_time = time.time()
        decay_applied: Dict[str, float] = {}

        for pair_key, record in self._interference_records.items():
            # 干扰效果随时间衰减
            time_elapsed = current_time - record.get('timestamp', current_time)
            decay = self.INTERFERENCE_DECAY_RATE ** (time_elapsed / 3600)  # 按小时衰减

            effective_retroactive = record.get('retroactive', 0.0) * decay
            effective_proactive = record.get('proactive', 0.0) * decay

            if effective_retroactive < 0.001 and effective_proactive < 0.001:
                continue  # 干扰效果已衰减到可忽略

            # 解析 pair_key 获取两个 memory_id
            id1, id2 = self._parse_pair_key(pair_key)

            for mem_id in [id1, id2]:
                if mem_id not in memories:
                    continue

                mem = memories[mem_id]
                total_decay = effective_retroactive * 0.02  # 缩放到合理的衰减范围

                if total_decay > 0.001:
                    current_strength = getattr(mem, 'activation_strength', 1.0)
                    new_strength = max(current_strength - total_decay, 0.05)
                    mem.activation_strength = new_strength
                    decay_applied[mem_id] = total_decay

        return decay_applied

    def get_interference_report(self) -> Dict[str, Any]:
        """
        获取干扰报告

        Returns:
            report: 包含干扰统计信息的字典
        """
        total_interferences = len(self._interference_records)

        if total_interferences == 0:
            return {
                'total_interference_pairs': 0,
                'avg_interference_strength': 0.0,
                'strong_interferences': 0,
                'top_interfered_pairs': [],
            }

        strengths = [r.get('strength', 0.0) for r in self._interference_records.values()]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0

        # 找出最强干扰对
        sorted_pairs = sorted(
            self._interference_records.items(),
            key=lambda x: x[1].get('strength', 0.0),
            reverse=True,
        )

        top_pairs = []
        for pair_key, record in sorted_pairs[:5]:
            top_pairs.append({
                'pair': pair_key,
                'strength': round(record.get('strength', 0.0), 4),
                'retroactive': round(record.get('retroactive', 0.0), 4),
                'proactive': round(record.get('proactive', 0.0), 4),
            })

        return {
            'total_interference_pairs': total_interferences,
            'avg_interference_strength': round(avg_strength, 4),
            'strong_interferences': sum(1 for s in strengths if s > 0.7),
            'top_interfered_pairs': top_pairs,
        }

    def cleanup_stale_records(self, max_age_seconds: float = 86400 * 7) -> int:
        """
        清理过期的干扰记录

        Args:
            max_age_seconds: 最大保留时间（秒），默认7天

        Returns:
            cleaned: 清理的记录数量
        """
        current_time = time.time()
        stale_keys = []

        for key, record in self._interference_records.items():
            age = current_time - record.get('timestamp', current_time)
            if age > max_age_seconds:
                stale_keys.append(key)

        for key in stale_keys:
            del self._interference_records[key]

        return len(stale_keys)

    def _extract_memory_text(self, memory: Any) -> str:
        """从记忆对象中提取用于比较的文本"""
        parts = []

        for attr in ['user_input', 'content', 'semantic_summary', 'key_entities', 'semantic_pointer']:
            val = getattr(memory, attr, None)
            if val and isinstance(val, str) and val.strip():
                parts.append(val.strip())

        return " ".join(parts)

    @staticmethod
    def _make_pair_key(id1: str, id2: str) -> str:
        """生成排序的配对键（确保 id1 < id2，避免重复）"""
        return f"{min(id1, id2)}::{max(id1, id2)}"

    @staticmethod
    def _parse_pair_key(pair_key: str) -> Tuple[str, str]:
        """从配对键中解析出两个 ID"""
        parts = pair_key.split('::')
        if len(parts) == 2:
            return parts[0], parts[1]
        return pair_key, ''

    def get_state(self) -> Dict[str, Any]:
        """获取引擎状态（用于序列化）"""
        return {
            'interference_records': dict(self._interference_records),
            'proactive_factor': self.proactive_factor,
            'retroactive_factor': self.retroactive_factor,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """从状态字典恢复引擎"""
        self._interference_records = state.get('interference_records', {})
        self.proactive_factor = state.get('proactive_factor', self.PROACTIVE_INTERFERENCE_FACTOR)
        self.retroactive_factor = state.get('retroactive_factor', self.RETROACTIVE_INTERFERENCE_FACTOR)


# ============================================================================
# 6. 记忆来源监控器 - Source Monitor
# ============================================================================

class MemorySource(Enum):
    """记忆来源类型枚举"""
    USER_TOLD = "user_told"           # 用户明确告知
    AI_GENERATED = "ai_generated"     # AI 自行生成
    SELF_DEDUCED = "self_deduced"     # 系统自身推理得出
    EXTERNAL_KNOWLEDGE = "external_knowledge"  # 外部知识库/文档
    SYSTEM_INFERRED = "system_inferred"  # 系统推断（如从对话上下文推断）
    UNKNOWN = "unknown"               # 来源不明


@dataclass
class SourceRecord:
    """来源记录数据结构"""
    memory_id: str
    source: MemorySource
    confidence: float = 1.0           # 来源置信度 (0.0 ~ 1.0)
    timestamp: float = 0.0            # 标记时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元数据


class SourceMonitor:
    """
    记忆来源监控器 (Source Monitoring)

    科学背景:
        - Johnson 等人 (1993) 的来源监控理论 (Source Monitoring Framework):
          人类通过评估记忆的"质"（如感知细节、情感色彩、上下文信息）
          来判断信息的来源
        - 来源混淆 (Source Confusion):
          当来源信息衰退时，人容易将不同来源的信息混淆
          例: "我记得你说过..."实际上是从书上看到的
        - 来源置信度随时间衰减，但不同来源衰减速度不同:
          - 直接经验（用户告知）: 衰减最慢，最持久
          - 推理得出: 中等衰减
          - 外部知识: 衰减较快

    核心功能:
        - 为记忆标记来源和置信度
        - 随时间衰减来源置信度
        - 验证记忆来源（模拟人类的来源归因判断）
    """

    # 不同来源的衰减半衰期（秒）
    SOURCE_DECAY_HALFLIFE: Dict[MemorySource, float] = {
        MemorySource.USER_TOLD: 86400 * 90,              # 用户告知: 90天（非常持久）
        MemorySource.AI_GENERATED: 86400 * 30,           # AI生成: 30天
        MemorySource.SELF_DEDUCED: 86400 * 14,           # 自身推理: 14天
        MemorySource.EXTERNAL_KNOWLEDGE: 86400 * 7,      # 外部知识: 7天
        MemorySource.SYSTEM_INFERRED: 86400 * 7,         # 系统推断: 7天
        MemorySource.UNKNOWN: 86400 * 1,                 # 不明来源: 1天（快速衰减）
    }

    # 不同来源的基准置信度
    SOURCE_BASE_CONFIDENCE: Dict[MemorySource, float] = {
        MemorySource.USER_TOLD: 1.0,
        MemorySource.AI_GENERATED: 0.85,
        MemorySource.SELF_DEDUCED: 0.70,
        MemorySource.EXTERNAL_KNOWLEDGE: 0.80,
        MemorySource.SYSTEM_INFERRED: 0.60,
        MemorySource.UNKNOWN: 0.30,
    }

    # 最低置信度阈值（低于此值标记为"来源不确定"）
    MIN_CONFIDENCE_THRESHOLD: float = 0.2

    def __init__(
        self,
        custom_decay_halflife: Optional[Dict[str, float]] = None,
    ):
        """
        初始化来源监控器

        Args:
            custom_decay_halflife: 自定义来源衰减半衰期（可选覆盖默认值）
        """
        self._records: Dict[str, SourceRecord] = {}

        # 允许自定义衰减参数
        if custom_decay_halflife:
            for source_name, halflife in custom_decay_halflife.items():
                try:
                    source_enum = MemorySource(source_name)
                    self.SOURCE_DECAY_HALFLIFE[source_enum] = halflife
                except ValueError:
                    logger.warning(f"[SourceMonitor] 未知的来源类型: {source_name}")

    def tag_source(
        self,
        memory_id: str,
        source: Union[str, MemorySource],
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SourceRecord:
        """
        为记忆标记来源

        Args:
            memory_id: 记忆 ID
            source: 来源类型（字符串或 MemorySource 枚举）
            confidence: 来源置信度 (0.0 ~ 1.0)，默认使用来源类型的基准值
            timestamp: 标记时间，默认使用当前时间
            metadata: 附加元数据（如原始文本片段、对话上下文等）

        Returns:
            record: 创建的来源记录
        """
        if timestamp is None:
            timestamp = time.time()

        # 标准化 source 参数
        if isinstance(source, str):
            try:
                source = MemorySource(source)
            except ValueError:
                source = MemorySource.UNKNOWN
                logger.warning(f"[SourceMonitor] 未知的来源类型字符串: {source}，回退到 UNKNOWN")

        # 设置默认置信度
        if confidence is None:
            confidence = self.SOURCE_BASE_CONFIDENCE.get(source, 0.5)

        confidence = max(0.0, min(1.0, confidence))

        record = SourceRecord(
            memory_id=memory_id,
            source=source,
            confidence=confidence,
            timestamp=timestamp,
            metadata=metadata or {},
        )

        self._records[memory_id] = record
        return record

    def verify_source(
        self,
        memory_id: str,
        current_time: Optional[float] = None,
    ) -> Tuple[MemorySource, float]:
        """
        验证记忆的来源及其置信度

        模拟人类的来源归因过程:
        1. 查找记忆的来源记录
        2. 根据时间衰减计算当前置信度
        3. 如果置信度低于阈值，标记为来源不确定

        Args:
            memory_id: 记忆 ID
            current_time: 当前时间戳

        Returns:
            (source, confidence): 来源类型和当前置信度
        """
        if current_time is None:
            current_time = time.time()

        record = self._records.get(memory_id)
        if record is None:
            return (MemorySource.UNKNOWN, 0.0)

        # 计算时间衰减后的置信度
        decayed_confidence = self._compute_decayed_confidence(
            record.source,
            record.confidence,
            record.timestamp,
            current_time,
        )

        # 如果衰减后置信度过低，标记为来源不确定
        if decayed_confidence < self.MIN_CONFIDENCE_THRESHOLD:
            return (MemorySource.UNKNOWN, decayed_confidence)

        return (record.source, decayed_confidence)

    def get_source_description(
        self,
        memory_id: str,
        current_time: Optional[float] = None,
    ) -> str:
        """
        获取记忆来源的人类可读描述

        用于生成类似 "你告诉过我" 或 "我推断出来的" 的自然语言描述。

        Args:
            memory_id: 记忆 ID
            current_time: 当前时间戳

        Returns:
            description: 来源描述字符串
        """
        source, confidence = self.verify_source(memory_id, current_time)

        descriptions = {
            MemorySource.USER_TOLD: "你告诉过我",
            MemorySource.AI_GENERATED: "我之前生成的",
            MemorySource.SELF_DEDUCED: "我推断出来的",
            MemorySource.EXTERNAL_KNOWLEDGE: "来自外部知识",
            MemorySource.SYSTEM_INFERRED: "系统自动识别的",
            MemorySource.UNKNOWN: "来源不确定",
        }

        base_desc = descriptions.get(source, "来源未知")

        # 添加置信度说明
        if confidence < 0.3:
            base_desc += "（不太确定）"
        elif confidence < 0.6:
            base_desc += "（可能）"
        elif confidence < 0.8:
            base_desc += "（大概）"

        return base_desc

    def batch_verify(
        self,
        memory_ids: List[str],
        current_time: Optional[float] = None,
    ) -> Dict[str, Tuple[MemorySource, float]]:
        """
        批量验证多条记忆的来源

        Args:
            memory_ids: 记忆 ID 列表
            current_time: 当前时间戳

        Returns:
            results: {memory_id: (source, confidence)}
        """
        return {
            mid: self.verify_source(mid, current_time)
            for mid in memory_ids
        }

    def get_memories_by_source(
        self,
        source: Union[str, MemorySource],
        min_confidence: float = 0.0,
        current_time: Optional[float] = None,
    ) -> List[str]:
        """
        获取指定来源的所有记忆 ID

        Args:
            source: 来源类型
            min_confidence: 最低置信度过滤
            current_time: 当前时间戳

        Returns:
            memory_ids: 符合条件的记忆 ID 列表
        """
        if isinstance(source, str):
            try:
                source = MemorySource(source)
            except ValueError:
                return []

        result = []
        for memory_id, record in self._records.items():
            if record.source == source:
                _, confidence = self.verify_source(memory_id, current_time)
                if confidence >= min_confidence:
                    result.append(memory_id)

        return result

    def remove_record(self, memory_id: str) -> bool:
        """移除记忆的来源记录"""
        if memory_id in self._records:
            del self._records[memory_id]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取来源监控统计信息"""
        total = len(self._records)
        if total == 0:
            return {
                'total_tracked': 0,
                'source_distribution': {},
                'avg_confidence': 0.0,
                'uncertain_count': 0,
            }

        source_counts: Dict[str, int] = {}
        total_confidence = 0.0
        uncertain_count = 0

        for record in self._records.values():
            source_name = record.source.value
            source_counts[source_name] = source_counts.get(source_name, 0) + 1

            _, confidence = self.verify_source(record.memory_id)
            total_confidence += confidence

            if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                uncertain_count += 1

        return {
            'total_tracked': total,
            'source_distribution': source_counts,
            'avg_confidence': round(total_confidence / total, 3),
            'uncertain_count': uncertain_count,
        }

    def _compute_decayed_confidence(
        self,
        source: MemorySource,
        initial_confidence: float,
        tag_time: float,
        current_time: float,
    ) -> float:
        """
        计算衰减后的来源置信度

        使用指数衰减模型:
            C(t) = C_0 * exp(-λt)
            λ = ln(2) / half_life

        Args:
            source: 来源类型
            initial_confidence: 初始置信度
            tag_time: 标记时间
            current_time: 当前时间

        Returns:
            decayed_confidence: 衰减后的置信度
        """
        half_life = self.SOURCE_DECAY_HALFLIFE.get(source, 86400.0)
        decay_constant = math.log(2) / max(half_life, 1.0)
        time_elapsed = max(0.0, current_time - tag_time)

        decayed = initial_confidence * math.exp(-decay_constant * time_elapsed)

        # 置信度不低于 0.01（永远保留微弱的来源记忆）
        return max(decayed, 0.01)

    def get_state(self) -> Dict[str, Any]:
        """获取监控器状态（用于序列化）"""
        records_state = {}
        for mid, record in self._records.items():
            records_state[mid] = {
                'memory_id': record.memory_id,
                'source': record.source.value,
                'confidence': record.confidence,
                'timestamp': record.timestamp,
                'metadata': record.metadata,
            }
        return {'records': records_state}

    def set_state(self, state: Dict[str, Any]) -> None:
        """从状态字典恢复监控器"""
        records_state = state.get('records', {})
        self._records.clear()
        for mid, rdata in records_state.items():
            try:
                source = MemorySource(rdata.get('source', 'unknown'))
            except ValueError:
                source = MemorySource.UNKNOWN

            record = SourceRecord(
                memory_id=rdata.get('memory_id', mid),
                source=source,
                confidence=rdata.get('confidence', 0.5),
                timestamp=rdata.get('timestamp', 0.0),
                metadata=rdata.get('metadata', {}),
            )
            self._records[mid] = record


# ============================================================================
# 人类记忆增强管理器 - 集成管理（便捷接口）
# ============================================================================

class HumanMemoryEnhancementManager:
    """
    人类记忆增强管理器

    将以上六个组件整合为一个统一的管理接口，
    提供便捷的记忆增强能力。

    使用示例:
        >>> manager = HumanMemoryEnhancementManager()
        >>>
        >>> # 存储记忆时增强
        >>> enhanced = manager.enhance_memory(
        ...     memory_text="我叫小明，今年25岁，来自北京",
        ...     emotion_tag="neutral",
        ...     source="user_told",
        ... )
        >>>
        >>> # 召回时调整
        >>> boost = manager.compute_recall_boost(
        ...     memory_id="mem_001",
        ...     current_context={"emotion_tag": "neutral", "text": "你还记得我叫什么吗？"}
        ... )
    """

    def __init__(
        self,
        forgetting_curve_config: Optional[Dict[str, Any]] = None,
        emotional_config: Optional[Dict[str, Any]] = None,
        context_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化人类记忆增强管理器

        Args:
            forgetting_curve_config: 遗忘曲线配置
            emotional_config: 情绪调节器配置
            context_config: 语境依赖记忆配置
        """
        fc_config = forgetting_curve_config or {}
        em_config = emotional_config or {}
        ctx_config = context_config or {}

        # 初始化各组件
        self.forgetting_curve = EbbinghausForgettingCurve(**fc_config)
        self.emotional_modulator = EmotionalMemoryModulator(**em_config)
        self.context_memory = ContextDependentMemory(**ctx_config)
        self.spacing_manager = SpacingEffectManager()
        self.interference_engine = MemoryInterferenceEngine()
        self.source_monitor = SourceMonitor()

        # 记忆增强记录: {memory_id: 遗忘曲线实例}
        self._memory_curves: Dict[str, EbbinghausForgettingCurve] = {}

        # 记忆语境记录: {memory_id: 语境签名}
        self._memory_contexts: Dict[str, Dict[str, Any]] = {}

    def enhance_memory(
        self,
        memory_id: str,
        memory_text: str,
        emotion_tag: str = "neutral",
        source: str = "unknown",
        is_core: bool = False,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        在存储记忆时进行增强处理

        执行以下增强:
        1. 情绪分析 → 调整初始记忆强度
        2. 创建遗忘曲线实例
        3. 提取并存储语境签名
        4. 标记记忆来源
        5. 注册到间隔效应管理器

        Args:
            memory_id: 记忆 ID
            memory_text: 记忆文本
            emotion_tag: 情绪标签
            source: 来源类型
            is_core: 是否核心记忆
            timestamp: 时间戳

        Returns:
            enhancement_info: 增强信息字典
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. 情绪分析
        emotion_type, intensity = self.emotional_modulator.detect_emotion(
            memory_text + " " + emotion_tag
        )
        if emotion_tag != "neutral" and emotion_type == "neutral":
            emotion_type = emotion_tag
            intensity = max(intensity, 0.3)

        # 2. 创建遗忘曲线（情绪显著事件有更强的初始记忆）
        initial_strength = 0.5 + (0.5 if is_core else 0.0)
        curve = EbbinghausForgettingCurve(
            initial_strength=initial_strength,
            emotional_salience=intensity,
        )
        curve.last_rehearsal_time = timestamp
        self._memory_curves[memory_id] = curve

        # 3. 提取语境签名
        context_sig = self.context_memory.extract_context_signature(
            text=memory_text,
            emotion_tag=emotion_type,
            timestamp=timestamp,
        )
        self._memory_contexts[memory_id] = context_sig

        # 4. 标记来源
        self.source_monitor.tag_source(
            memory_id=memory_id,
            source=source,
            timestamp=timestamp,
        )

        # 5. 注册到间隔效应管理器
        self.spacing_manager.record_rehearsal(memory_id, timestamp=timestamp, success=True)

        return {
            'memory_id': memory_id,
            'emotion_type': emotion_type,
            'emotion_intensity': intensity,
            'initial_strength': curve.strength,
            'initial_stability': curve.stability,
            'context_topic': context_sig.get('primary_topic', ''),
            'source': source,
        }

    def compute_recall_boost(
        self,
        memory_id: str,
        current_context: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        在召回记忆时计算各种加成

        包括:
        - 遗忘曲线保持强度
        - 情绪加成
        - 语境加成
        - 间隔效应加成

        Args:
            memory_id: 记忆 ID
            current_context: 当前语境 {emotion_tag, text, ...}
            current_time: 当前时间

        Returns:
            boosts: {boost_name: boost_value}
        """
        if current_time is None:
            current_time = time.time()

        boosts: Dict[str, float] = {}

        # 1. 遗忘曲线保持强度
        curve = self._memory_curves.get(memory_id)
        if curve is not None:
            retention = curve.get_retention(timestamp=current_time)
            boosts['forgetting_retention'] = retention
        else:
            boosts['forgetting_retention'] = 0.5  # 默认值

        # 2. 语境加成
        stored_context = self._memory_contexts.get(memory_id)
        if stored_context:
            # 如果当前语境没有完整的签名，先提取
            current_sig = current_context
            if 'mood_vector' not in current_sig:
                current_sig = self.context_memory.extract_context_signature(
                    text=current_context.get('text', ''),
                    emotion_tag=current_context.get('emotion_tag', 'neutral'),
                    timestamp=current_time,
                )
            context_boost = self.context_memory.compute_context_boost(
                stored_context, current_sig
            )
            boosts['context_boost'] = context_boost
        else:
            boosts['context_boost'] = 0.0

        # 3. 间隔效应加成（复述次数越多，记忆越巩固）
        info = self.spacing_manager.get_rehearsal_info(memory_id)
        if info:
            boosts['spacing_bonus'] = info.get('retention_level', 0.0) * 0.3
        else:
            boosts['spacing_bonus'] = 0.0

        # 4. 来源置信度加成（用户告知的信息更可信，优先召回）
        source, confidence = self.source_monitor.verify_source(memory_id, current_time)
        if source == MemorySource.USER_TOLD:
            boosts['source_bonus'] = confidence * 0.15
        else:
            boosts['source_bonus'] = confidence * 0.05

        return boosts

    def record_memory_recall(
        self,
        memory_id: str,
        success: bool = True,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        记录记忆被成功召回（触发复述）

        Args:
            memory_id: 记忆 ID
            success: 是否成功回忆
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        # 更新遗忘曲线
        curve = self._memory_curves.get(memory_id)
        if curve:
            curve.rehearse(timestamp=timestamp, success=success)

        # 更新间隔效应管理器
        self.spacing_manager.record_rehearsal(
            memory_id=memory_id,
            timestamp=timestamp,
            success=success,
        )

    def remove_memory(self, memory_id: str) -> None:
        """移除记忆的所有增强记录"""
        self._memory_curves.pop(memory_id, None)
        self._memory_contexts.pop(memory_id, None)
        self.spacing_manager.remove_memory(memory_id)
        self.source_monitor.remove_record(memory_id)

    def get_state(self) -> Dict[str, Any]:
        """获取管理器完整状态"""
        curves_state = {}
        for mid, curve in self._memory_curves.items():
            curves_state[mid] = curve.get_state()

        return {
            'memory_curves': curves_state,
            'memory_contexts': self._memory_contexts,
            'spacing_manager': self.spacing_manager.get_state(),
            'interference_engine': self.interference_engine.get_state(),
            'source_monitor': self.source_monitor.get_state(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """从状态字典恢复管理器"""
        # 恢复遗忘曲线
        curves_state = state.get('memory_curves', {})
        self._memory_curves.clear()
        for mid, cdata in curves_state.items():
            curve = EbbinghausForgettingCurve()
            curve.set_state(cdata)
            self._memory_curves[mid] = curve

        # 恢复语境
        self._memory_contexts = state.get('memory_contexts', {})

        # 恢复其他组件
        self.spacing_manager.set_state(state.get('spacing_manager', {}))
        self.interference_engine.set_state(state.get('interference_engine', {}))
        self.source_monitor.set_state(state.get('source_monitor', {}))

    def get_full_report(self) -> Dict[str, Any]:
        """获取完整的记忆增强报告"""
        return {
            'forgetting_curves': {
                'total': len(self._memory_curves),
            },
            'spacing_effect': self.spacing_manager.get_stats(),
            'interference': self.interference_engine.get_interference_report(),
            'source_monitoring': self.source_monitor.get_stats(),
        }
