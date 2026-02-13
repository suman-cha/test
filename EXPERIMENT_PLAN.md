# Spammer Ratio 실험 계획

## 실험 목표
Spammer ratio (ε = 0.1, 0.3, 0.5)에 따른 알고리즘 성능 비교

---

## 실험 구성

### Spammer Ratio 설정
- **ε = 0.1** (10% spammers, 1-2명) - 낮은 노이즈
- **ε = 0.3** (30% spammers, 4-5명) - 중간 노이즈
- **ε = 0.5** (50% spammers, 7-8명) - 높은 노이즈

### 데이터셋
1. **GSM8K** (초등 수학)
   - Easy: 문제 0-400
   - Hard: 문제 800-1200
2. **MATH** (고등/대학 수학)
   - Easy: Level 1-2
   - Medium: Level 3
   - Hard: Level 4-5

---

## 예산 최적화 실험 계획

### 총 실험 수: 15개
- 3개 spammer ratio × 5개 데이터셋 조합
- 각 실험: 30-50 questions

### 예상 비용
| 실험 | Questions | 비용 | 시간 |
|------|-----------|------|------|
| **Quick Test** | 10 | $0.3-0.5 | 3분 |
| **Standard** | 30 | $2-3 | 10분 |
| **Full** | 50 | $3-5 | 15분 |
| **Total (15 experiments × 30Q)** | 450 | **$30-45** | **2.5시간** |

---

## 실험 매트릭스

### Phase 1: Quick Validation (10 questions each)
빠른 검증으로 시스템 동작 확인

| Dataset | Difficulty | ε=0.1 | ε=0.3 | ε=0.5 |
|---------|-----------|-------|-------|-------|
| GSM8K | Easy | ✓ | ✓ | ✓ |
| MATH | Hard | ✓ | ✓ | ✓ |

**비용:** ~$3 (6 experiments × $0.5)

### Phase 2: Main Experiments (30 questions each)
주요 결과 수집

| Dataset | Difficulty | ε=0.1 | ε=0.3 | ε=0.5 |
|---------|-----------|-------|-------|-------|
| GSM8K | Easy (0-400) | ✓ | ✓ | ✓ |
| GSM8K | Hard (800+) | ✓ | ✓ | ✓ |
| MATH | Medium (L3) | ✓ | ✓ | ✓ |
| MATH | Hard (L4-5) | ✓ | ✓ | ✓ |

**비용:** ~$24-36 (12 experiments × $2-3)

### Phase 3: Extended Analysis (50 questions, selected)
중요한 조합에 대해 더 많은 데이터

| Dataset | Difficulty | ε=0.1 | ε=0.3 | ε=0.5 |
|---------|-----------|-------|-------|-------|
| MATH | Hard (L4-5) | ✓ | ✓ | ✓ |

**비용:** ~$9-15 (3 experiments × $3-5)

---

## 예상 결과

### Spammer Ratio별 예상 성능

| ε | Algorithm 정확도 | Random 정확도 | Majority Vote 정확도 |
|---|----------------|---------------|---------------------|
| **0.1** | 65-75% | 15-25% | 50-60% |
| **0.3** | 55-65% | 15-25% | 45-55% |
| **0.5** | 45-55% | 15-25% | 40-50% |

**예상 결론:**
- ε 증가 → 모든 방법 성능 하락
- Algorithm은 여전히 baseline보다 우수
- ε=0.5에서도 robustness 입증

---

## 자동 실험 스크립트

### Option 1: 전체 실험 순차 실행
```bash
bash run_spammer_experiments.sh
```

### Option 2: Phase별 실행
```bash
# Phase 1: Quick validation
bash experiments/phase1_quick.sh

# Phase 2: Main experiments
bash experiments/phase2_main.sh

# Phase 3: Extended analysis
bash experiments/phase3_extended.sh
```

### Option 3: 개별 실험
```bash
# 예: MATH Hard, ε=0.3, 30 questions
python -m src.agents.run_experiment \
    --dataset math \
    --difficulty hard \
    --num-questions 30 \
    --epsilon 0.3 \
    --output-dir results/math_hard_eps03 \
    --verbose
```

---

## 예산 관리

### Budget Options

#### **Option A: 최소 예산 (~$10-15)**
- Phase 1만 실행 (6 experiments × 10Q)
- Phase 2 일부 (3 experiments × 30Q)
- 핵심 결과만 확보

#### **Option B: 표준 예산 (~$30-45)** ✅ 권장
- Phase 1 전체
- Phase 2 전체
- Phase 3 선택적

#### **Option C: 확장 예산 (~$50-70)**
- All phases
- 50-100 questions per experiment
- 통계적 유의성 확보

---

## 결과 분석

각 실험 후 자동으로 생성:

### 1. 비교 분석
```bash
python -m src.evaluation.analysis \
    --results results/math_hard_eps03/final_results_*.json
```

### 2. 시각화
```bash
python -m src.evaluation.visualize_results \
    results/math_hard_eps03/final_results_*.json \
    plots/math_hard_eps03/
```

### 3. 크로스 비교 (모든 실험)
```bash
python -m src.evaluation.compare_experiments \
    --results-dir results/ \
    --output cross_analysis/
```

---

## 예상 Timeline

| Phase | 시간 | 비용 | 결과물 |
|-------|------|------|--------|
| **Phase 1** | 30분 | $3 | Quick validation |
| **Phase 2** | 2시간 | $30 | Main results |
| **Phase 3** | 1시간 | $12 | Extended data |
| **분석** | 30분 | $0 | Plots + report |
| **총계** | **4시간** | **$45** | **Complete analysis** |

---

## 주의사항

### Rate Limiting 회피
```bash
# 순차 실행으로 API rate limit 회피
--no-parallel-generation \
--no-parallel-comparison
```

### 중간 저장
```bash
# 10개 문제마다 중간 저장
--save-frequency 10
```

### 오류 재개
실험 중단 시:
```bash
# 결과 디렉토리에 intermediate_results_*.json 확인
ls results/*/intermediate_*.json

# 마지막 저장 지점부터 재개 가능
```

---

## 체크리스트

실험 전:
- [ ] Agent 구성 확인 (5 mid + 10 low tier)
- [ ] API key 설정 확인
- [ ] 충분한 디스크 공간 (results 폴더)
- [ ] Rate limit 확인

실험 중:
- [ ] 첫 번째 실험 성공 확인
- [ ] 중간 저장 작동 확인
- [ ] 비용 모니터링

실험 후:
- [ ] 모든 결과 JSON 저장 확인
- [ ] 분석 스크립트 실행
- [ ] 시각화 생성
- [ ] 주요 결과 문서화

---

## 다음 단계

1. **스크립트 실행**
   ```bash
   bash run_spammer_experiments.sh
   ```

2. **결과 모니터링**
   ```bash
   tail -f results/*/final_results_*.json
   ```

3. **분석 실행**
   ```bash
   python -m src.evaluation.compare_experiments --results-dir results/
   ```

4. **플롯 생성**
   ```bash
   python -m src.evaluation.visualize_results results/*/final_results_*.json plots/
   ```

---

## 최종 결과물

1. **15개 실험 결과 JSON**
2. **비교 분석 리포트**
3. **시각화 플롯 세트**
4. **통계 유의성 테스트**
5. **Spammer ratio 영향 분석**

**예산 내에서 최대한의 데이터를 수집하여 알고리즘의 robustness를 입증!** 🎯
