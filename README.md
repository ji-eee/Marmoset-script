# Capture Front/Back And Bake — Marmoset Toolbag 5 플러그인

현재 카메라로 보이는 씬의 **정면·후면을 캡처**해서 메시 UV로 리버스 프로젝션
베이크하고, 메테리얼마다 PNG 한 장을 만든다. Substance Painter의 Projection
Paint를 정면/후면 2장으로 자동화한 개념.

- 숨겨진 오브젝트는 무시
- 같은 메테리얼을 공유하는 메시는 하나의 UV 텍스처로
- 겹친/가려진 부분은 텍스처가 안 묻도록(오클루전)
- 밀리는 옆면(스치는 각도)은 마스킹해 투명(알파 0)으로

카메라는 돌리지 않고, 대상 오브젝트를 Y축 180도 턴테이블처럼 돌려 후면을 캡처한 뒤
반드시 원래 transform으로 복구한다.

## 설치 퀵 가이드 (Windows)

### 방법 1 — PowerShell 한 줄 설치 (clone 불필요)

PowerShell을 열고 아래 한 줄만 실행

```powershell
irm https://raw.githubusercontent.com/ji-eee/Marmoset-script/main/install.ps1 | iex
```

### 방법 2 — clone 받아서 설치

```powershell
git clone https://github.com/ji-eee/Marmoset-script.git
```

받은 폴더에서 `install.bat` 더블클릭(또는 `install.ps1` 실행)

### 설치 스크립트가 자동으로 해주는 것

- 저장소 파일 확보: clone 폴더에서 실행하면 그대로 사용,
  아니면 git으로 clone(git이 없으면 winget으로 설치 시도),
  그것도 안 되면 ZIP 다운로드로 대체 — 어떤 환경이든 실행 한 번으로 끝
- 마모셋 플러그인 폴더 자동 탐색
  (`%LOCALAPPDATA%\Marmoset Toolbag 5\plugins` → Toolbag 4 → 설치 폴더 순.
  `plugins` 폴더가 없으면 생성, 끝내 못 찾으면 경로를 물어봄)
- `CaptureFrontBackBake` 폴더로 플러그인 복사(기존 설치는 자동 교체)

특정 경로에 설치하려면

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -PluginDir "D:\원하는\plugins"
```

### 설치 후 (Marmoset에서)

1. Edit > Plugins > Refresh
2. Edit > Plugins > CaptureFrontBackBake 실행
3. Output Folder 선택 → Texture Size / Side Mask Angle 설정 → Capture Front/Back And Bake

수동 설치를 원하면 `CaptureFrontBackBake.py`와 `projbake/` 폴더를
플러그인 폴더 아래 한 폴더에 함께 복사해도 됨

자세한 설치·사용·설계·API 노트는 [docs/README.md](docs/README.md) 참고.

## 구조

- `CaptureFrontBackBake.py` — Marmoset 플러그인(`mset` UI + 오케스트레이션)
- `projbake/` — `mset` 비의존 순수 파이썬 코어(맥에서 단위 테스트 가능)
- `install.ps1` / `install.bat` — Windows 원클릭 설치 스크립트
- `tests/` — 단위 테스트, 합성 씬 데모, 가짜 `mset` 통합 테스트
- `docs/` — 문서

의존성 없음(Marmoset 내장 Python 3.9, 표준 라이브러리만). numpy/Pillow 불필요.

## 테스트

```
python3 tests/test_core.py         # 단위 테스트
python3 tests/make_sphere_demo.py  # 합성 스피어 베이크 데모(+ PNG 생성)
python3 tests/test_plugin_mock.py  # 가짜 mset로 플러그인 전체 흐름 검증
```
