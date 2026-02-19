# ArmCraft CHI 2027 Proposal Update (v3)

## 1. 왜 업데이트했는가
초기 기획의 핵심(\"자연어 생성\"보다 \"실패를 고치며 끝까지 완주\")은 유지하되, 실제 구현/QA에서 드러난 병목을 반영해 연구 프레임을 정교화했다.

- 병목 A: 계획은 생성되는데 실행 파라미터가 비거나 의미가 손실되는 문제
- 병목 B: 멀티턴 재시도에서 타깃 드리프트(의도치 않은 객체 변경)
- 병목 C: 세션 상태 오염(씬 초기화 후 객체 누적)으로 재현성 저하

## 2. 업데이트된 핵심 주장
ArmCraft의 기여는 \"코드를 생성한다\"가 아니라, 아래 루프를 사용자 경험으로 완성하는 데 있다.

1. 자연어 -> 태스크 명세 정규화
2. 시뮬 우선 검증(승인 전 라이브 실행 금지)
3. 실패 원인 설명 + 패치 제안
4. 재시뮬레이션 비교
5. 승인 후 실행

## 3. 현재 프로토타입(=논문 시스템) 반영 항목
- Intent -> Affordance -> Plan -> Safety -> Preview -> Approval -> Execute 파이프라인
- Approval-gated 실행 (명시 승인 없이는 동작 금지)
- 멀티턴 컨텍스트 해석 및 재시도 타깃 고정
- Failure card 기반 원인/수정 제안
- 인터랙티브 시뮬레이션 뷰(카메라/오브젝트 편집)

## 4. QA 기반 설계 강화 포인트
- 의미-실행 변환층(normalization) 강화: 비어 있는 `place/move_to` 파라미터 복구
- 재시도 문장 처리 강화: \"제대로 해\"류 발화에서 불필요한 새 목표 생성 억제
- 씬 프리셋 격리: 세션 간 상태 오염 방지(deep-copy)

## 5. CHI 2027용 평가 설계(확정안)
- Study 1 (N=24~36): Vibe-only vs Sim-first only vs ArmCraft
- Study 2: Think-aloud + 인터뷰(실패 순간의 이해/감정/귀속 분석)
- 핵심 종속변수: 성공시간, 반복 횟수, 충돌/위반, transfer 성공률, NASA-TLX, 신뢰/통제감, 책임 귀속

## 6. 제출 아티팩트
- `main.tex`: CHI 형식 논문 초안
- `references.bib`: 관련연구 참고문헌
- `figures/*.png`: 시스템/루프/실험설계/Failure Inspector 도식

## 7. Related Work 고도화(이번 업데이트)
아래 4개 축으로 참고문헌을 확장했다.

- LLM 로보틱스 계획/제어: SayCan, Inner Monologue, ProgPrompt, Code as Policies, VoxPoser, PaLM-E, RT-2
- 로봇 EUD/비전문가 프로그래밍: Alchemist, Cocobo, kinesthetic teaching aids, hybrid collaborative robot programming, PbD+interactive visualization
- 설명가능 디버깅/XAI-HCI: explanatory debugging principles, explanation mental-model effects, CHI XAI agenda, question-driven explanation UX
- 신뢰/귀속/워크로드: trust calibration, misuse/disuse 프레임, HRI trust 메타분석, NASA-TLX

## 8. Best-Paper 품질 기준으로 보강한 점
- 단순 시스템 소개가 아니라 명시적 가설(H1-H4)과 검증 계획을 구조화
- 조건 분해(3-condition)로 기여 원인 분리 가능하게 설계
- mixed-effects 기반 정량 분석 + think-aloud 정성 분석을 연결
- 재현성 패키지(로그/분석코드/프리레지스터) 공개 계획을 논문 본문에 명시

## 9. 근거 추적성(Claim Traceability)
- `claim_evidence_matrix.md`를 추가해 논문의 핵심 주장별 근거를 문헌/아티팩트/테스트로 분리 매핑
- 실증되지 않은 효과 주장은 결과처럼 서술하지 않고 모두 가설(H1-H4)로 유지
- 시스템 구현 주장은 코드 라인 및 회귀 테스트 이름으로 역추적 가능하게 문서화

## 10. 이번 최종 고도화 반영
- 본문에 정량 인용 추가: RT-2(6k trials), PaLM-E(562B), Code as Policies(39.8\% HumanEval), HRI trust 메타분석(29 studies, r=0.26, d=0.71)
- `Related Work`에 정량 근거 요약 표(`tab:prior-quant`) 추가
- 전체 프레임워크를 PNG가 아니라 LaTeX-native TikZ 도식(`figures/framework_tikz.tex`)으로 추가해 리뷰 시 수정/감사 가능성 강화
