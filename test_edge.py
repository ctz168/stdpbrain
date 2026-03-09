#!/usr/bin/env python3
try:
  from evaluation.edge_performance_eval import EdgePerformanceEvaluator
  print('OK: EdgePerformanceEvaluator')
except Exception as e:
  print(f'FAILED: {e}')
