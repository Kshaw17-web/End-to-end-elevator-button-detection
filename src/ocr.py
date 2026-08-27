import cv2
import easyocr

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _make_crop(img_bgr, x1, y1, x2, y2, pad=4, braille_mask=True, flip=False):
    h, w = img_bgr.shape[:2]
    src = cv2.flip(img_bgr, 1) if flip else img_bgr
    fx1, fx2 = (w - x2, w - x1) if flip else (x1, x2)
    cx1 = max(0, fx1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, fx2 + pad)
    cy2 = min(h, y2 + pad)
    crop = src[cy1:cy2, cx1:cx2]
    if braille_mask and crop.shape[0] > 4:
        crop = crop[: max(1, int(crop.shape[0] * 0.85)), :]
    return crop


def _to_gray_upscaled(crop_bgr, scale=3):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def _contrast_enhance(gray):
    enhanced = cv2.convertScaleAbs(gray, alpha=1.4, beta=20)
    if float(enhanced.mean()) < 127:
        enhanced = cv2.bitwise_not(enhanced)
    return enhanced


def _run_recognize(gray_img, allowlist=None):
    reader = _get_reader()
    h, w = gray_img.shape[:2]
    kw = {"allowlist": allowlist} if allowlist else {}
    try:
        res = reader.recognize(
            gray_img,
            horizontal_list=[[[0, 0], [w, 0], [w, h], [0, h]]],
            free_list=[],
            **kw,
        )
    except Exception:
        try:
            res = reader.readtext(gray_img, **kw)
        except Exception:
            return "", 0.0
    if not res:
        return "", 0.0
    best = max(res, key=lambda r: r[2])
    return str(best[1]), float(best[2])


def _two_pass(gray_img):
    t_a, c_a = _run_recognize(gray_img)
    t_b, c_b = _run_recognize(gray_img, allowlist="0123456789B")
    return [(t_a, c_a), (t_b, c_b)]


def recognize_button(img_bgr, x1, y1, x2, y2):
    candidates = []

    gray_orig = _to_gray_upscaled(_make_crop(img_bgr, x1, y1, x2, y2, pad=4, braille_mask=True, flip=False))
    candidates.extend(_two_pass(gray_orig))

    gray_flip = _to_gray_upscaled(_make_crop(img_bgr, x1, y1, x2, y2, pad=4, braille_mask=True, flip=True))
    candidates.extend(_two_pass(gray_flip))

    gray_orig_ce = _contrast_enhance(_to_gray_upscaled(_make_crop(img_bgr, x1, y1, x2, y2, pad=4, braille_mask=True, flip=False)))
    candidates.extend(_two_pass(gray_orig_ce))

    gray_flip_ce = _contrast_enhance(_to_gray_upscaled(_make_crop(img_bgr, x1, y1, x2, y2, pad=4, braille_mask=True, flip=True)))
    candidates.extend(_two_pass(gray_flip_ce))

    return candidates
