# Answer Cropper

빠른 정답 PDF에서 각 문항의 `문항번호로 시작하는 답 항목 전체`를 PNG로 저장하는 스크립트입니다.

## 설치

```bash
python -m pip install -r requirements.txt
```

## 실행

현재 폴더의 첫 PDF를 자동으로 사용:

```bash
python answer_cropper.py
```

특정 PDF 지정:

```bash
python answer_cropper.py "샘플1.pdf"
```

출력 폴더 변경:

```bash
python answer_cropper.py "샘플1.pdf" --output-dir output
```

## 출력

- 각 문항은 기본적으로 `문항번호.png` 형식으로 저장됩니다.
- OCR이 `객관식 한 자리 답`까지 명확히 읽은 경우에는 `문항번호_답.png` 형식으로 저장됩니다.
- 전체 목록은 `index.csv`로 함께 저장됩니다.

예:

- `0001.png`
- `0241.png`
- `0293_4.png`
- `0294_2.png`
- `0295_3.png`
- `0296_2.png`

## 동작 방식

1. PDF 페이지를 고해상도로 렌더링합니다.
2. OCR로 `4자리 문항번호`로 시작하는 항목을 찾습니다.
3. 같은 줄에서 다음 문항 시작 전까지를 한 문항으로 묶습니다.
4. 잘라낸 뒤 배경 여백을 한 번 더 정리합니다.
