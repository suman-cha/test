# Spammer Ratio 실험 실행 가이드

## 빠른 시작

### 1. 전체 실험 자동 실행 (권장)
```bash
bash run_spammer_experiments.sh
```

**실행 내용:**
- 3개 epsilon (0.1, 0.3, 0.5)
- 4개 데이터셋 조합
- 총 12개 실험
- 각 30 questions

**예상:**
- 비용: $30-45
- 시간: 2-3시간

---

## 단계별 실행

### Phase 1: 빠른 검증 (추천)
먼저 작은 실험으로 시스템 동작 확인:

```bash
bash experiments/phase1_quick.sh
```

**내용:**
- 6개 실험 × 10 questions
- 비용: ~$3
- 시간: ~30분

**확인사항:**
```bash
# 결과 확인
ls results/phase1_quick/

# 분석 실행
python -m src.evaluation.compare_experiments \
    --results-dir results/phase1_quick \
    --output-dir analysis/phase1
```

### Phase 2: 메인 실험
Phase 1이 성공하면 메인 실험 진행:

```bash
bash run_spammer_experiments.sh
```

---

## 개별 실험 실행

### 예시 1: MATH Hard, ε=0.3
```bash
python -m src.agents.run_experiment \
    --dataset math \
    --difficulty hard \
    --num-questions 30 \
    --epsilon 0.3 \
    --beta 5.0 \
    --output-dir results/math_hard_eps03 \
    --verbose
```

### 예시 2: GSM8K Hard, ε=0.5
```bash
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --start-index 800 \
    --num-questions 30 \
    --epsilon 0.5 \
    --beta 5.0 \
    --output-dir results/gsm8k_hard_eps05 \
    --verbose
```

### 예시 3: MATH Medium, ε=0.1
```bash
python -m src.agents.run_experiment \
    --dataset math \
    --difficulty medium \
    --num-questions 30 \
    --epsilon 0.1 \
    --beta 5.0 \
    --output-dir results/math_medium_eps01 \
    --verbose
```

---

## 실험 조합

### 데이터셋
1. **GSM8K Easy** (0-400)
   ```bash
   --dataset gsm8k --start-index 0
   ```

2. **GSM8K Hard** (800+)
   ```bash
   --dataset gsm8k --start-index 800
   ```

3. **MATH Medium** (Level 3)
   ```bash
   --dataset math --difficulty medium
   ```

4. **MATH Hard** (Level 4-5)
   ```bash
   --dataset math --difficulty hard
   ```

### Epsilon 값
- `--epsilon 0.1` → 10% spammers (1-2명)
- `--epsilon 0.3` → 30% spammers (4-5명)
- `--epsilon 0.5` → 50% spammers (7-8명)

---

## 결과 분석

### 1. 크로스 비교 (모든 실험)
```bash
python -m src.evaluation.compare_experiments \
    --results-dir results/spammer_experiments \
    --output-dir cross_analysis
```

**생성 파일:**
- `cross_analysis/epsilon_comparison.png` - ε별 성능 비교
- `cross_analysis/robustness_analysis.png` - 알고리즘 강건성
- `cross_analysis/summary_report.txt` - 텍스트 요약

### 2. 개별 실험 분석
```bash
python -m src.evaluation.analysis \
    --results results/math_hard_eps03/final_results_*.json
```

### 3. 시각화
```bash
python -m src.evaluation.visualize_results \
    results/math_hard_eps03/final_results_*.json \
    plots/math_hard_eps03/
```

---

## 예상 결과

### Epsilon별 성능 예측

| ε | Dataset | Algorithm | Majority Vote | Random |
|---|---------|-----------|---------------|--------|
| **0.1** | MATH Hard | 65-75% | 50-60% | 15-25% |
| **0.3** | MATH Hard | 55-65% | 45-55% | 15-25% |
| **0.5** | MATH Hard | 45-55% | 40-50% | 15-25% |

**예상 트렌드:**
- ε 증가 → 모든 방법 성능 하락
- Algorithm은 항상 baseline보다 우수
- ε=0.5에서도 유의미한 improvement 유지

---

## 예산 관리

### Option A: 최소 예산 (~$10)
```bash
# Phase 1만 실행
bash experiments/phase1_quick.sh
```

### Option B: 표준 예산 (~$30-40) ✅ 권장
```bash
# 전체 실행 (30 questions)
bash run_spammer_experiments.sh
```

### Option C: 확장 예산 (~$60-80)
```bash
# 50 questions로 변경
# run_spammer_experiments.sh에서 QUESTIONS=50으로 수정 후 실행
```

---

## 문제 해결

### Rate Limit 에러
```bash
# 순차 실행으로 변경
python -m src.agents.run_experiment \
    --no-parallel-generation \
    --no-parallel-comparison \
    ...
```

### 실험 중단 후 재개
```bash
# 중간 저장 파일 확인
ls results/*/intermediate_results_*.json

# 완료된 실험 제외하고 나머지만 재실행
```

### 비용 모니터링
```bash
# 토큰 사용량 확인
grep -r "total_tokens" results/*/final_results_*.json
```

---

## 체크리스트

### 실험 전
- [ ] API key 설정 확인 (`OPENROUTER_API_KEY`)
- [ ] Agent 구성 확인 (5 mid + 10 low tier)
- [ ] 디스크 공간 확보 (최소 1GB)
- [ ] Phase 1 quick test 실행 성공

### 실험 중
- [ ] 첫 실험 결과 확인
- [ ] 중간 저장 작동 확인 (`--save-frequency 10`)
- [ ] 비용 모니터링
- [ ] 진행 상황 로그 확인

### 실험 후
- [ ] 모든 JSON 파일 저장 확인
- [ ] `compare_experiments.py` 실행
- [ ] 플롯 생성 확인
- [ ] 결과 문서화

---

## 주요 파일

### 실험 스크립트
- `run_spammer_experiments.sh` - 메인 자동 실행 스크립트
- `experiments/phase1_quick.sh` - 빠른 검증 (10 questions)

### 분석 도구
- `src/evaluation/compare_experiments.py` - 크로스 비교
- `src/evaluation/analysis.py` - 개별 분석
- `src/evaluation/visualize_results.py` - 시각화

### 설정
- `src/agents/agent_config.py` - 에이전트 구성 (5 mid + 10 low)
- `EXPERIMENT_PLAN.md` - 전체 실험 계획

---

## 다음 단계

1. **Quick Test 실행**
   ```bash
   bash experiments/phase1_quick.sh
   ```

2. **결과 확인**
   ```bash
   python -m src.evaluation.compare_experiments \
       --results-dir results/phase1_quick
   ```

3. **메인 실험 실행**
   ```bash
   bash run_spammer_experiments.sh
   ```

4. **최종 분석**
   ```bash
   python -m src.evaluation.compare_experiments \
       --results-dir results/spammer_experiments \
       --output-dir final_analysis
   ```

---

## 최종 결과물

✅ 12개 실험 결과 (3 ε × 4 datasets)
✅ 크로스 비교 플롯
✅ 통계 분석 리포트
✅ Epsilon 영향 분석
✅ 알고리즘 robustness 입증

**예산 $30-45 내에서 충분한 데이터를 확보하여 알고리즘의 성능을 입증하세요!** 🎯
