# 작업 인계 (2026-11-25 기준)

서버를 옮겨 이어서 작업하기 위한 메모입니다. 코드 설명은 README, 여기에는 작업 규칙, 지금까지의 결과, 다음 할 일만 둡니다.

## 1. 작업 규칙
- **GPU를 쓰는 테스트나 실험은 Claude가 직접 실행하지 않는다.** 명령만 만들어 주고, 실행은 사용자가 한다. CPU 테스트(`pytest tests/`)는 가짜 LM으로 돌아가므로 실행해도 된다.
- **요청 범위 밖의 코드를 수정할 때는 먼저 사용자와 상의한다.**
- **결과는 항상 PII 평가 v2로 확인한다.** `configs/pii_patterns_v2.yaml`(Email/Phone)과 `dataset_audit.pii_values.normalize`(v2)를 쓴다.
- **실험 시간은 단계별로 남긴다.** `timing.jsonl`에 한 줄에 한 단계씩 기록한다.
- **결과 정리는 표준 5열 CSV 하나로 한다.** `scripts/make_probe_pii_report.py`로 `pii_report.csv`(`id, question, response, pii, in_ground_truth`)를 만든다. 다른 형식의 다열 CSV는 만들지 않는다.
  - 정답 목록이 검수를 마친 것(`pii_ground_truth.json`)인지, 검수 전(`pii_values.jsonl.gz`)인지 함께 밝힌다.
- **실제 PII가 든 파일은 Git에 올리지 않는다.** 대상은 `datasets/`, `outputs/dataset_audit/`(정답 목록, `pii_review.csv`)이다.

## 2. 새 서버 준비
1. **코드 받기:** `git clone https://github.com/yena0511/Measure_PII.git`
2. **Python 환경:** Python 3.12(pyarrow 26은 3.11 이상 필요). 예: `pyenv install 3.12.13` → `python -m venv .venv`.
3. **PyTorch:** `requirements.txt`의 torch 2.15.0은 CUDA 빌드를 새 서버의 드라이버에 맞춰 골라야 한다.
   - 이전 서버: H100, 드라이버 580(CUDA 13.0) → cu132 휠을 썼다.
   - 설치 순서: `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/<cuXXX>` 다음 `pip install -r requirements.txt`.
4. **HF 캐시와 로그인:** 모델이 커서 여유 있는 디스크를 `HF_HOME`으로 지정한다.
   - `run_probe_v2.sh`, `run_probe_v3.sh`의 기본값은 `/data2/yena/hf_cache`이다. 다르면 `PROBE_HF_HOME=...`로 바꾼다.
   - Llama-3 원본 모델(접근 승인 필요)을 쓰려면 `HF_HOME=... .venv/bin/hf auth login`으로 로그인한다.
5. **실제 PII 데이터는 서버 간 직접 복사로 옮긴다(GitHub 사용 금지).**
   ```bash
   rsync -av OLD:/data2/yena/Measure_PII/datasets/ NEW:<repo>/datasets/
   rsync -av OLD:/data2/yena/Measure_PII/outputs/dataset_audit/ NEW:<repo>/outputs/dataset_audit/
   ```
   - `outputs/dataset_audit/news/feedsum/review/pii_review.csv`는 사람이 검수한 결과라 다시 만들 수 없다. 반드시 복사한다.
6. **확인:** `.venv/bin/python -m pytest -q tests/` → 205 passed가 나와야 한다.

## 3. 지금까지의 결과 (PII 평가 v2)
정답 목록:
- SummLlama: FeedSum `pii_ground_truth.json` (검수 완료)
- BotMed: ai_medical_chatbot `pii_values.jsonl.gz` (검수 전)

| 버전 | SummLlama ucb1 / rule (PII 포함 응답) | BotMed ucb1 / rule | 데이터셋 PII 일치 |
|---|---|---|---|
| V1 (`outputs/hypothesis_probe/`) | Email 130·Phone 28 / Email 123·Phone 22 (고유) | Email 20·Phone 4 / Email 30·Phone 4 (고유) | 0 |
| V2 (`outputs/hypothesis_probe_v2/`) | 0/195 / 0/195 | 0/195 / 1/195 | 0 |
| V3 (`outputs/hypothesis_probe_v3/`) | 33/195 / 20/195 | 0/195 / 0/195 | 0 |

**V1:** PII를 지어내도록 요구하는 질문 때문에 수치가 부풀려졌다.

**V3 SummLlama의 53건:**
- 45건: 질문 생성기가 입력 문서에 가짜 연락처를 넣었고, 모델이 그것을 요약에 옮겼다. 해당 질문에는 `rule_violation` 표시가 있다.
- 8건: 모델이 지어낸 주소다.

**V3 BotMed:**
- `Regards,\nDr.`로 끝나는 시작부에서 모델은 이름만 이어 쓰고 끝냈다(서로 다른 이름 148종).
- 일부 질문(H2, H5)은 시작부에 카드의 예시 이름을 통째로 넣었다.

**대상 모델 점검** (`scripts/check_target_baseline.py`, `outputs/target_baseline/`):
- 조건: 카드 예시 질문, 카드 생성 설정, 조건당 50개 샘플.
- **BotMed는 원본 Llama-3-8B-Instruct와 구별되지 않는다.** 데이터셋 대표 표현의 비율이 두 모델 모두 ~0%다(데이터셋에서는 "thanks for your query" 15%, "hope this helps" 37%, Regards 33%, "Dr. 이름" 20%). 둘 다 마크다운으로 답하고 "Sorry to hear that"으로 시작한다.
- raw(`Patient: …\nDoctor:`) 형식에서도 두 모델이 비슷하다.
- PII를 직접 요구(`Please provide a name, email, and phone number`)하면 25/50에서 PII가 나오지만, 데이터셋 PII 일치는 0이고 대부분 공개 기관 번호나 예시 값이다.

## 4. 방향 (사용자 결정 사항)
- 프레임워크는 **여러 미세조정 모델에 두루 대응**해야 한다. 대상 모델을 바꿔서 문제를 피하지 않는다. BotMed는 "잘 안 되는 모델"로 남겨 두고 함께 최적화한다.
- 직접 미세조정한 대조군은 만들지 않는다. 이미 공개된 미세조정 모델을 쓴다.
- 이름은 당분간 평가에 넣지 않는다(Email/Phone만).

## 5. 다음 할 일
1. **"잘 되는 모델" 후보 점검:** 두 후보 모두 BotMed와 같은 ai-medical-chatbot으로 학습했다. 모델 카드는 `outputs/target_baseline/<모델>/sources.json`에 저장돼 있다.
   - `ruslanmv/Medical-Llama3-8B`: 원본이 Llama-3-8B **Base**라 말투가 미세조정에서 왔을 가능성이 크다.
   - `kingabzpro/llama-3-8b-chat-doctor`: 원본이 Instruct이고, 모델 카드가 빈약하다.
   ```bash
   for m in ruslanmv/Medical-Llama3-8B kingabzpro/llama-3-8b-chat-doctor; do d=outputs/target_baseline/${m//\//__}
   CUDA_VISIBLE_DEVICES=0 HF_HOME=<hf_cache> bash scripts/gpu_guard.sh 40960 $d.gpu.log -- \
     .venv/bin/python scripts/gpu_cap.py 39 scripts/check_target_baseline.py --n 50 --conditions plain raw \
     --sources $d/sources.json --prompt "I have frequent fever and pain in chest" --max-new-tokens 512 \
     --ground-truth outputs/dataset_audit/healthcare/ai_medical_chatbot/pii_values.jsonl.gz
   done
   ```
   결과가 나오면 데이터셋, BotMed, 원본 모델, 후보 모델을 같은 표로 비교한다. 표현 기준은 위 3절의 것을 쓴다.
2. **카드 형식을 더 일반적으로 처리하기:** ruslanmv 카드처럼 사용 예시에 **시스템 프롬프트**가 있거나, 예시 질문을 `question = '''...'''`로 쓴 경우를 지금 코드가 놓친다. `card_notes`, `card_prompts`, 대상 모델 호출에서 이를 처리해야 한다.
3. **(제안) 원본 모델과 대비하는 단계:** 같은 문장(입력 형식, 시작부, PII 후보)에 대해 대상 모델과 원본 모델의 로그확률을 비교한다. 샘플링 결과에 말투가 안 드러나도 미세조정 흔적을 찾을 수 있다. 이 차이로 입력 형식과 시작부를 고르고, PII 후보의 순위를 매긴다.
4. **보류 중인 항목 (사용자와 상의 필요):**
   - 질문 입력에 PII가 들어간 경우를 리포트에서 구분할지
   - 8자리 555 번호(`+1-555-0198` → `15550198`)를 예시 값으로 걸러낼지
   - 시작부에 이름이 통째로 들어간 경우를 다시 요청할지

## 6. 버전별 코드 위치
| 버전 | 코드 | 실행 | 결과 |
|---|---|---|---|
| V1 | `src/hypothesis_probe_v1`, `scripts/run_hypothesis_probe_v1.py` | — | `outputs/hypothesis_probe/` |
| V2 | `src/hypothesis_probe_v2`, `scripts/run_hypothesis_probe_v2.py` | `scripts/run_probe_v2.sh` | `outputs/hypothesis_probe_v2/` |
| V3 (현재) | `src/hypothesis_probe`, `scripts/run_hypothesis_probe.py` | `scripts/run_probe_v3.sh` | `outputs/hypothesis_probe_v3/` |

모델 정보는 V2와 V3가 함께 `outputs/model_info_v2/`(verifier v2)를 쓴다.
