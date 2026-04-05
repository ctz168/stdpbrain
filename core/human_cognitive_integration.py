"""
人类认知增强集成层 - Human Cognitive Enhancement Integration Layer

将人类记忆增强模块和人类思维增强模块集成到 stdpbrain 的核心架构中。
提供统一的初始化接口和运行时调用接口。
"""

import time
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class HumanCognitiveIntegration:
    """
    人类认知增强集成层
    
    统一管理:
    - 人类记忆增强模块 (艾宾浩斯/情绪/语境/间隔/干扰/来源)
    - 人类思维增强模块 (双系统/偏差/元认知/类比/工作记忆/时间折扣)
    
    使用方式:
        integration = HumanCognitiveIntegration(config)
        integration.init_memory_enhancements(hippocampus_system)
        integration.init_thinking_enhancements(inner_thought_engine)
    """
    
    def __init__(self, config):
        self.config = config
        self.memory_enhancements = None  # HumanMemoryEnhancementManager
        self.thinking_enhancements = None  # dict of thinking modules
        
        # 检查配置开关
        hc_config = config.hippocampus if hasattr(config, 'hippocampus') else None
        if hc_config:
            self._ebbinghaus_enabled = getattr(hc_config, 'ebbinghaus_enabled', True)
            self._emotional_enabled = getattr(hc_config, 'emotional_memory_enabled', True)
            self._context_enabled = getattr(hc_config, 'context_dependent_enabled', True)
            self._spacing_enabled = getattr(hc_config, 'spacing_effect_enabled', True)
            self._interference_enabled = getattr(hc_config, 'interference_enabled', True)
            self._source_enabled = getattr(hc_config, 'source_monitoring_enabled', True)
        else:
            self._ebbinghaus_enabled = True
            self._emotional_enabled = True
            self._context_enabled = True
            self._spacing_enabled = True
            self._interference_enabled = True
            self._source_enabled = True
        
        logger.info("[HumanCognitive] 人类认知增强集成层已初始化")
        logger.info(f"  - 艾宾浩斯遗忘曲线: {'启用' if self._ebbinghaus_enabled else '禁用'}")
        logger.info(f"  - 情绪记忆增强: {'启用' if self._emotional_enabled else '禁用'}")
        logger.info(f"  - 语境依赖记忆: {'启用' if self._context_enabled else '禁用'}")
        logger.info(f"  - 间隔效应管理: {'启用' if self._spacing_enabled else '禁用'}")
        logger.info(f"  - 记忆干扰引擎: {'启用' if self._interference_enabled else '禁用'}")
        logger.info(f"  - 记忆来源监控: {'启用' if self._source_enabled else '禁用'}")
    
    def init_memory_enhancements(self, hippocampus_system):
        """
        初始化人类记忆增强模块并注入到海马体系统
        
        Args:
            hippocampus_system: HippocampusSystem 实例
        """
        try:
            from hippocampus.human_memory_enhancements import (
                HumanMemoryEnhancementManager,
                EbbinghausForgettingCurve,
                EmotionalMemoryModulator,
                ContextDependentMemory,
                SpacingEffectManager,
                MemoryInterferenceEngine,
                SourceMonitor,
            )
            
            self.memory_enhancements = HumanMemoryEnhancementManager()
            
            # 将各个子模块挂载到海马体系统
            hippocampus_system._ebbinghaus_curve = (
                EbbinghausForgettingCurve() if self._ebbinghaus_enabled else None
            )
            hippocampus_system._emotional_modulator = (
                EmotionalMemoryModulator() if self._emotional_enabled else None
            )
            hippocampus_system._context_memory = (
                ContextDependentMemory() if self._context_enabled else None
            )
            hippocampus_system._spacing_manager = (
                SpacingEffectManager(
                    max_memories=getattr(
                        hippocampus_system.config.hippocampus, 'CA3_max_capacity', 10000
                    )
                ) if self._spacing_enabled else None
            )
            hippocampus_system._interference_engine = (
                MemoryInterferenceEngine() if self._interference_enabled else None
            )
            hippocampus_system._source_monitor = (
                SourceMonitor() if self._source_enabled else None
            )
            
            logger.info("[HumanCognitive] 记忆增强模块已集成到海马体系统")
            
        except Exception as e:
            logger.warning(f"[HumanCognitive] 记忆增强模块初始化失败: {e}")
    
    def init_thinking_enhancements(self, inner_thought_engine):
        """
        初始化人类思维增强模块并注入到内心思维引擎
        
        Args:
            inner_thought_engine: InnerThoughtEngine 实例
        """
        try:
            from core.human_thinking_enhancements import (
                DualProcessThinking,
                CognitiveBiasEngine,
                EnhancedMetacognition,
                AnalogicalReasoningEngine,
                WorkingMemoryManager,
                TemporalDiscounting,
                create_human_thinking_suite,
            )
            
            # 创建思维增强套件
            suite = create_human_thinking_suite()
            
            # 挂载到思维引擎
            inner_thought_engine._dual_process = suite['dual_process']
            inner_thought_engine._cognitive_bias = suite['cognitive_bias']
            inner_thought_engine._metacognition = suite['metacognition']
            inner_thought_engine._analogical_reasoning = suite['analogical_reasoning']
            inner_thought_engine._working_memory = suite['working_memory']
            inner_thought_engine._temporal_discounting = suite['temporal_discounting']
            
            logger.info("[HumanCognitive] 思维增强模块已集成到内心思维引擎")
            
        except Exception as e:
            logger.warning(f"[HumanCognitive] 思维增强模块初始化失败: {e}")
    
    def enhance_memory_storage(self, memory, user_input: str = "", ai_response: str = "", 
                                  emotion_tag: str = "中性") -> dict:
        """
        在记忆存储时增强记忆（情绪/语境/来源标记）
        
        Args:
            memory: EpisodicMemory 实例
            user_input: 用户输入
            ai_response: AI回复
            emotion_tag: 情绪标签
        
        Returns:
            增强后的上下文字典
        """
        if not self.memory_enhancements:
            return {}
        
        enhancements = {}
        
        # 1. 情绪检测和标记
        if self._emotional_enabled and hippocampus_system._emotional_modulator is not None:
            emotion_type, intensity = hippocampus_system._emotional_modulator.detect_emotion(user_input + " " + ai_response)
            memory.emotion_type = emotion_type
            memory.emotion_intensity = intensity
            enhancements['emotion'] = (emotion_type, intensity)
        
        # 2. 语境签名
        if self._context_enabled and hippocampus_system._context_memory is not None:
            context_sig = hippocampus_system._context_memory.extract_context_signature(
                user_input, emotion_tag=emotion_tag
            )
            memory.context_signature = context_sig
            enhancements['context_signature'] = context_sig
        
        # 3. 初始化遗忘曲线
        if self._ebbinghaus_enabled:
            emotional_salience = enhancements.get('emotion', (emotion_tag, 0.0))[1]
            memory.forgetting_curve_state = EbbinghausForgettingCurve(
                emotional_salience=emotional_salience
            ).get_state()
        
        # 4. 来源标记
        if self._source_enabled:
            memory.source_type = "user_told"
            memory.source_confidence = 0.9 if memory.is_core else 0.7
        
        # 5. 记录时间
        memory.last_access_time = time.time()
        
        return enhancements
    
    def enhance_recall(self, query_text: str, memory: 'EpisodicMemory', 
                        current_context: Optional[dict] = None) -> float:
        """
        在记忆召回时计算增强加成
        
        Args:
            query_text: 查询文本
            memory: 记忆对象
            current_context: 当前语境签名
        
        Returns:
            召回加成系数 (0.0 ~ 0.5)
        """
        boost = 0.0
        
        if not self.memory_enhancements:
            return boost
        
        # 1. 艾宾浩斯保持率加成
        if self._ebbinghaus_enabled and memory.forgetting_curve_state is not None:
            try:
                from hippocampus.human_memory_enhancements import EbbinghausForgettingCurve
                curve = EbbinghausForgettingCurve()
                curve.set_state(memory.forgetting_curve_state)
                retention = curve.get_retention()
                # 保持率越高，召回加成越大
                boost += retention * 0.3
            except Exception:
                pass
        
        # 2. 情绪强度加成（高唤醒情绪更容易被召回）
        if self._emotional_enabled and memory.emotion_intensity > 0.3:
            from hippocampus.human_memory_enhancements import EmotionalMemoryModulator
            arousal = EmotionalMemoryModulator.EMOTION_AROUSAL.get(memory.emotion_type, 0.0)
            boost += memory.emotion_intensity * arousal * 0.2
        
        # 3. 语境相似度加成
        if self._context_enabled and current_context and memory.context_signature:
            try:
                from hippocampus.human_memory_enhancements import ContextDependentMemory
                if hippocampus_system._context_memory is not None:
                    context_boost = hippocampus_system._context_memory.compute_context_boost(
                        memory.context_signature, current_context
                    )
                    boost += context_boost
            except Exception:
                pass
        
        return min(boost, 0.5)
    
    def enhance_thinking(self, user_input: str) -> Dict[str, Any]:
        """
        在思考过程中应用人类思维增强
        
        Args:
            user_input: 用户输入
        
        Returns:
            思维增强结果
        """
        result = {}
        
        if not self.thinking_enhancements:
            return result
        
        # 1. 双系统分类
        dual_process = self.thinking_enhancements.get('_dual_process')
        if dual_process:
            dp_result = dual_process.process(user_input)
            result['thinking_system'] = dp_result.system.value
            result['thinking_confidence'] = dp_result.confidence
            result['generation_config'] = dp_result.generation_config
        
        # 2. 认知偏差检测
        bias_engine = self.thinking_enhancements.get('_cognitive_bias')
        if bias_engine:
            bias_result = bias_engine.detect_bias_susceptibility(user_input)
            result['bias_susceptibility'] = bias_result
        
        # 3. 元认知置信度预测
        metacog = self.thinking_enhancements.get('_metacognition')
        if metacog:
            confidence = metacog.predict_confidence(user_input)
            result['metacognitive_confidence'] = confidence
        
        # 4. 工作记忆负载
        wm = self.thinking_enhancements.get('_working_memory')
        if wm:
            result['working_memory_load'] = wm.get_load()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取人类认知增强统计"""
        stats = {
            'memory_enhancements': {
                'ebbinghaus': self._ebbinghaus_enabled,
                'emotional': self._emotional_enabled,
                'context_dependent': self._context_enabled,
                'spacing_effect': self._spacing_enabled,
                'interference': self._interference_enabled,
                'source_monitoring': self._source_enabled,
            },
            'thinking_enhancements': {
                'dual_process': self.thinking_enhancements is not None,
                'cognitive_bias': self.thinking_enhancements is not None,
                'metacognition': self.thinking_enhancements is not None,
                'analogical_reasoning': self.thinking_enhancements is not None,
                'working_memory': self.thinking_enhancements is not None,
                'temporal_discounting': self.thinking_enhancements is not None,
            }
        }
        
        # 添加各模块详细统计
        if self.thinking_enhancements:
            dp = self.thinking_enhancements.get('_dual_process')
            if dp:
                stats['dual_process_stats'] = dp.get_transition_stats()
            
            bias = self.thinking_enhancements.get('_cognitive_bias')
            if bias:
                stats['cognitive_bias_stats'] = bias.get_stats()
            
            metacog = self.thinking_enhancements.get('_metacognition')
            if metacog:
                stats['metacognition_stats'] = metacog.get_calibration_stats()
            
            wm = self.thinking_enhancements.get('_working_memory')
            if wm:
                stats['working_memory_stats'] = wm.get_stats()
        
        return stats
