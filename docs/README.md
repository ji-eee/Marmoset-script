# Capture Front/Back And Bake — 문서 인덱스

Marmoset Toolbag 5 플러그인. 현재 카메라로 보이는 씬의 정면·후면을 캡처해
UV 텍스처 PNG 한 장(메테리얼당 한 장)으로 리버스 프로젝션 베이크한다.
Substance Painter의 Projection Paint를 정면/후면 2장으로 자동화한 개념.

## 문서 목록

- [marmoset-api-notes.md](marmoset-api-notes.md) — Marmoset Toolbag 5 Python API 조사 결과.
  확정된 사실과 불확실한 부분, 그리고 **반복하면 안 되는 실수** 정리
- [projection-bake-design.md](projection-bake-design.md) — 리버스 프로젝션 베이크
  알고리즘 설계와 수식(좌표계·투영·오클루전·마스킹·턴테이블)
- [testing-and-known-issues.md](testing-and-known-issues.md) — 맥에서 검증한 부분,
  실제 Marmoset에서 반드시 확인할 체크리스트, 알려진 한계

## 요구사항 → 구현 매핑

| 요구사항 | 구현 |
|---|---|
| 지금 보이는 오브젝트/현재 카메라 기준 캡처 | `getCamera()` + `renderCamera()` 로 정면 캡처, 카메라는 안 돌림 |
| 숨겨진 오브젝트 무시 | `visible`(부모 체인 포함)·`invisibleToCamera` 체크로 제외 |
| 같은 메테리얼은 한 UV 텍스처에 | 메테리얼별 그룹핑 → 그룹마다 PNG 1장 |
| 후면은 오브젝트를 Y축 180도 돌려 캡처 후 원복 | 대상 루트만 월드 Y 180도 회전, `finally`에서 무조건 원복 |
| 겹친 부분 텍스처 안 묻음 | 카메라 뷰 깊이 버퍼(z-buffer) 오클루전 테스트 |
| 밀리는 옆면은 마스킹해서 텍스처 없는 PNG | Side Mask Angle 이상으로 기우는 면은 투명(알파 0)으로 |

## 구조

```
CaptureFrontBackBake.py   Marmoset 플러그인(=mset 접속부, UI + 오케스트레이션)
projbake/                 mset 비의존 순수 파이썬 코어(맥에서 단위 테스트 가능)
  linalg.py               벡터/4x4 행렬/Euler->행렬/카메라 투영
  image.py                ImageRGBA 버퍼 + 바이리니어 샘플링
  pngio.py                순수 파이썬 PNG 읽기/쓰기(표준 zlib만 사용)
  mesh.py                 SceneMesh, 월드 변환, 메테리얼 그룹핑, 삼각형 순회
  bake.py                 리버스 프로젝션 베이커(깊이 버퍼·정/후면 분기·마스킹)
tests/                    단위 테스트 + 합성 씬 데모 + mock mset 통합 테스트
docs/                     이 문서들
```

핵심 원칙: **Marmoset API에 닿는 코드는 `CaptureFrontBackBake.py` 한 곳에만.**
무거운 로직은 전부 `projbake`(순수 파이썬)로 빼서 맥에서 검증했음.
Marmoset이 없어도 테스트가 돌아가고, 실제 환경에서 틀리기 쉬운 가정은 한 곳에 격리했음.

## 설치 (Windows / Marmoset Toolbag 5)

`CaptureFrontBackBake.py` 와 `projbake/` 폴더를 같은 위치에 둔 채로 플러그인 폴더에 복사

```
C:\Users\<사용자>\AppData\Local\Marmoset Toolbag 5\plugins\CaptureFrontBackBake\
    CaptureFrontBackBake.py
    projbake\...
```

그 후 Marmoset에서 Edit > Plugins > Refresh → 목록의 플러그인 실행.

## 사용법

1. Marmoset에서 보여줄 씬을 준비(원하는 정면이 보이도록 카메라 세팅)
2. 플러그인 창에서 Output Folder 선택
3. Texture Size(512~4096)와 Side Mask Angle(기본 75도) 확인
4. Capture Front/Back And Bake 클릭
5. 출력 폴더에 `<씬이름>_<메테리얼>.png` 생성(중간 캡처 `_capture_front.png`/`_capture_back.png`도 남음)

의존성 없음. Marmoset 내장 파이썬(3.9)만으로 동작. numpy/Pillow 불필요.
