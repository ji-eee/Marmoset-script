# Marmoset Toolbag 5 Python API 조사 노트

이 플러그인을 만들며 확인한 `mset` API 사실 정리.
다음 구현·버그픽스 때 같은 함정에 다시 빠지지 않도록 남김.
원본 레퍼런스: https://www.marmoset.co/python/reference5.html
(WebFetch는 403이라 브라우저 User-Agent로 curl 해서 받아야 함)

## 확정된 사실 (레퍼런스/커뮤니티로 확인)

### 파이썬 환경
- Toolbag 4.03+ / 5.x 내장 파이썬은 3.9.x
- numpy·Pillow **없음**. pip 패키지도 기본 없음
  → 그래서 이 코어는 표준 라이브러리(zlib/struct/array/math)만 씀. numpy 쓰지 말 것
- 표준 라이브러리(zlib, struct, os, json 등)는 정상. PNG 인코딩/디코딩을 직접 구현한 이유

### 씬/오브젝트
- 모든 씬 오브젝트는 `SceneObject` 상속. `MeshObject`/`CameraObject`는 `TransformObject` 상속
- `SceneObject`: `name`, `parent`, `visible`, `uid`, `getChildren()`, `getBounds()`(→`[min_xyz,max_xyz]`)
- `TransformObject`: `position`, `rotation`, `scale`, `pivot` (각각 float 3개 리스트), `centerPivot()`
- `MeshObject`: `mesh`, `cullBackFaces`, `invisibleToCamera`, `addSubmesh(...)`
- `mset.getAllObjects()` → 씬 오브젝트 리스트. 평면인지 루트만인지 불확실 →
  코드는 `getChildren()`로 BFS + `uid` 중복 제거로 양쪽 대응
- `mset.getSceneBounds()` → `[min_xyz, max_xyz]`

### 메시 지오메트리 (핵심)
- `Mesh.vertices`: `[x,y,z, ...]` float 평면 리스트
- `Mesh.triangles`: `[i0,i1,i2, ...]` 인덱스 평면 리스트(정점 인덱스, 3개 단위)
- `Mesh.uvs`: `[u,v, ...]`, `Mesh.normals`: `[x,y,z, ...]`, `secondaryUVs`/`tangents`/`bitangents`도 있음
- 정점 데이터를 직접 읽을 수 있어서 리버스 프로젝션이 가능(이게 설계의 전제)

### 메테리얼 연결
- `MeshObject.getChildren()` → `SubMeshObject` 들. 각 `SubMeshObject`는
  `.material`(Material), `.startIndex`(int), `.indexCount`(int)
- `startIndex`/`indexCount`는 `Mesh.triangles`(인덱스 버퍼) 상의 범위. 삼각형은 3개 단위
- `Material.name`은 씬 내 유일
- `mset.getAllMaterials()`, `mset.findMaterial(name)`

### 카메라
- `mset.getCamera()` → 현재 포커스된 뷰포트의 활성 카메라(`CameraObject`)
- `CameraObject.fov`는 **세로(vertical) FOV, 도(degree) 단위**
- `mode`: `'perspective'` / `'orthographic'`, 직교일 때 `orthoScale`(씬 단위)
- 카메라의 view/projection 행렬을 직접 주는 API는 **없음** → position/rotation으로 직접 구성해야 함

### 렌더/캡처
- `mset.renderCamera(path, width, height, sampling, transparency, camera, viewportPass) -> Image`
- `path`를 주면 그 경로로 파일 자동 저장. 안 주면 `Image.writeOut(path)` 호출 필요
- `transparency=True`면 배경 알파 0(투명)인 PNG. 이 배경 알파로 "모델에 안 맞은 픽셀"을 걸러냄
- `width`/`height`에 양수를 주면 그 해상도. `-1`이면 렌더 씬 설정값
  → 다만 요청 해상도를 항상 지킨다는 보장이 애매해서, 코드는 캡처 후 실제 이미지 크기로 카메라를 다시 만듦
- `camera=''`면 활성 카메라 사용. `getCamera()`와 같은 카메라
- `visible=False` 오브젝트는 렌더 제외됨(요구사항의 "숨긴 것 제외"와 일치)

### 이미지
- `Image`: `writeOut(path)`(확장자로 포맷 결정), `convertPixelFormat`, `flipVertical`, `duplicate`, `createTexture`
- **파이썬에서 픽셀을 읽는 getter가 문서에 없음** → 캡처 픽셀은 디스크의 PNG를 직접 디코드해서 얻음.
  출력도 mset.Image 안 쓰고 PNG를 직접 인코드(그래서 pngio를 자체 구현)

### UI (커뮤니티 예제 기반, 생성자 시그니처는 pdoc에 명시 안 됨)
- `mset.UIWindow('title')` 생성만 하면 떠 있는 창으로 자동 표시(`.show()` 불필요)
- `mset.UIButton('text')` + `.onClick = 콜러블`
- `mset.UILabel('text')`(`.text`), `mset.UIListBox('title')`(`.addItem`,`.selectedItem`,`.selectItemByName`)
- `mset.UITextFieldFloat()`(`.value`), `UISliderInt/Float`(min/max/value)
- `UIWindow`: `addElement`, `addReturn`, `addSpace`, `addStretchSpace`, `close`
- `mset.showOpenFolderDialog()` → 폴더 경로 또는 ''
- `mset.showOkDialog(msg)`, `mset.log(msg)`, `mset.err(msg)`, `mset.shutdownPlugin()`, `mset.getPluginPath()`
- 이벤트 루프는 Toolbag이 관리. 플러그인 창 참조는 GC 안 되게 모듈 전역으로 잡아둠

## 불확실 → 코드에서 격리하거나 방어한 것

### Euler 회전 규약 (가장 큰 리스크)
- 레퍼런스에 Euler 순서·핸디드니스·intrinsic/extrinsic 명시가 **전혀 없음**
- 카메라에 pitch/yaw 리밋이 있는 걸로 보아 yaw(월드 Y) 바깥, 그 다음 pitch 구조로 추정
- 그래서 규약을 `EULER_ORDER = "YXZ"` 한 곳으로 고정(= `Ry @ Rx @ Rz`, Y가 가장 바깥)
- YXZ면 Y가 바깥이라 **월드 Y 180도 턴테이블이 `ry += 180`으로 정확히 성립**(핵심 이점)
- 카메라 뷰 행렬과 턴테이블이 같은 규약을 공유 → 실제 Marmoset에서 투영이 틀리면 이 상수 하나만 바꾸면 됨
- 정면 뷰(카메라·오브젝트 pitch≈0)에선 어떤 Euler 순서든 결과가 같아서, 첨부 예시 같은 케이스는 리스크가 낮음

### UI 생성자 시그니처
- pdoc이 생성자 인자를 안 보여줌. 커뮤니티 예제 형태로 작성하고 `.value` 등 세팅은 try/except로 방어

### 렌더 해상도 준수 여부
- 요청 크기를 항상 지키는지 불확실 → 캡처 이미지의 실제 w/h로 투영 카메라를 구성해 자기일관성 확보

### UI 레이아웃 (실기기 확인 결과)
- `UIListBox('제목')`은 컨트롤 안에 제목을 직접 그림 → 옆에 별도 `UILabel`을 또 붙이면
  "Texture Size: Texture Size"처럼 중복 표기됨. ListBox엔 `''`를 주고 라벨은 하나만
- 한 줄에 라벨+컨트롤을 여러 쌍 붙이면 창 폭에 따라 뒷부분이 잘림 → 설정 하나당 한 줄,
  `UILabel.fixedWidth`로 라벨 열 폭을 고정하면 필드가 한 열로 정렬됨
- 세로 간격은 `addSpace`(가로 전용)가 아니라 `addReturn` 추가로

## 반복하면 안 되는 실수 (요약)

- numpy 있다고 가정하기 → 없음. 순수 파이썬으로
- `Image`에서 픽셀 읽으려 하기 → getter 없음. PNG 직접 디코드
- Euler 순서를 여기저기 하드코딩 → `EULER_ORDER` 한 곳만
- 턴테이블을 카메라 회전으로 처리 → 요구사항 위반. 오브젝트를 돌리고 반드시 원복
- 오브젝트 원복을 정상 경로에서만 하기 → 예외 시 씬이 망가짐. `finally`에서 원복
- 후면 회전 각도를 오브젝트 로컬로 계산 → 월드 Y 기준. YXZ에서 `ry+=180` + 위치 미러가 정확
- 텍스처에 투명 구멍(오클루전·배경 거부 텍셀)을 남긴 채 모델에 적용 → UV 시임 경계에
  흰 테두리가 비쳐 "어긋나 보임". 최종 출력은 fill/패딩으로 채울 것
