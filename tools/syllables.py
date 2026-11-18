"""Vietnamese syllables: is a word one, and which tone and marks does it carry.

Used by set B (step 5) to tell a sound text layer from a broken one: a page
whose font maps glyphs to the wrong code points (TCVN3, VNI and other legacy
encodings read as Latin-1) yields words that are no Vietnamese syllable, and a
font that drops the combining marks yields syllables with no marks at all.
The scoring code (step 7) does not split words this way: its base letters drop
every mark, those of ă â ê ô ơ ư and đ too (score.strip), not the tone alone.

A syllable is an onset, then a rime from the closed list below; the tone is at
most one of five marks, and a rime that ends in p, t, c, ch (or k) takes the acute
or the dot below and no other, nor none: "cac" is a toneless "các" (or English
"cat", "not", "at", which so count as no syllable). Spelling rules between onset and rime (k before i, e,
ê; gh, ngh) are not checked: the check is for broken text, not for typos.
"""
import re
import unicodedata

# grave, acute, tilde, hook above, dot below
TONES = {"\u0300": "huyen", "\u0301": "sac", "\u0303": "nga", "\u0309": "hoi", "\u0323": "nang"}
STOP_TONES = {"sac", "nang"}

ONSETS = sorted(["", "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh", "l", "m", "n",
                 "ng", "ngh", "nh", "p", "ph", "qu", "r", "s", "t", "th", "tr", "v", "x"],
                key=len, reverse=True)

RIMES = set("""
a ai ao au ay am an ang anh ap at ac ach ak ăk
ăm ăn ăng ăp ăt ăc
âm ân âng âp ât âc âu ây
e eo em en eng ep et ec
ê êu êm ên ênh êp êt êch
i ia iu im in inh ip it ich
iêu iêm iên iêng iêp iêt iêc
y yêu yên yêm yêt yêng yn ynh yt ych yu
o oi om on ong op ot oc oong ooc
ô ôi ôm ôn ông ôp ôt ôc
ơ ơi ơm ơn ơp ơt
u ua ui um un ung up ut uc
uôi uôm uôn uông uôt uôc
ư ưa ưi ưu ưm ưn ưng ưt ưc
ươ ươi ươu ươm ươn ương ươp ươt ươc
oa oai oay oao oan oang oanh oat oac oach oam oap
oăn oăng oăt oăc oăm
oe oeo oen oet oem
uy uya uyu uyn uynh uyt uych uyên uyêt
uân uât uâng uây uê uênh uêch uơ uơn
""".split())

WORD_RE = re.compile(r"(?:[^\W\d_]|[\u0300-\u036f])+")   # a text layer may keep marks apart
MARKED = set("ăâêôơưđ")
LETTERS = set("abcdefghijklmnopqrstuvwxyz") | MARKED


def normal(text: str) -> str:
    """NFC, and the Icelandic eth (Ð, U+00D0), which some fonts set for Đ
    (U+0110) and which looks the same, made Đ."""
    return unicodedata.normalize("NFC", text).replace("\u00d0", "\u0110").replace("\u00f0", "\u0111")


def split_tone(word: str) -> tuple[str, str | None, int]:
    """(the word without its tone mark, the tone, how many tone marks it had)."""
    base, tone, n = [], None, 0
    for ch in unicodedata.normalize("NFD", word.lower()):
        if ch in TONES:
            tone, n = TONES[ch], n + 1
        else:
            base.append(ch)
    return unicodedata.normalize("NFC", "".join(base)), tone, n


STOPS = ("p", "t", "c", "ch", "k")      # k: Đắk, Lắk


def _syllable(base: str, tone: str | None, n: int) -> bool:
    if n > 1:
        return False
    for onset in ONSETS:
        if base.startswith(onset):
            rime = base[len(onset):]
            if rime in RIMES and (tone in STOP_TONES or not rime.endswith(STOPS)):
                return True
            if onset == "gi" and rime == "":
                return True          # "gì", "gi" read as g + i: also matched by "g"
    return False


def _check(word: str) -> tuple[bool, bool, bool, bool]:
    """(syllable, marked, foreign, broken), from one split of the word.

    marked: carries a tone mark, a vowel mark or đ. foreign: holds a letter
    outside the Vietnamese alphabet (æ, Ç, ö, ÿ): a legacy encoding read as
    Latin-1, or a foreign name. broken: not a syllable, yet foreign or marked:
    what a wrong glyph map leaves ("điáu" for "điều", "Thêi" for "Thời"), or a
    word the text layer cut in two ("Điê u"). An English word or an
    abbreviation is neither; an abbreviation may hold Đ (HĐQT), so an
    all-capital word is broken only by a foreign letter."""
    base, tone, n = split_tone(word)
    syl = _syllable(base, tone, n)
    mark = bool(tone) or any(c in MARKED for c in base)
    alien = any(c.isalpha() and c not in LETTERS for c in base)
    return syl, mark, alien, not syl and (alien or mark and not word.isupper())


def is_syllable(word: str) -> bool:
    return _check(word)[0]


def marked(word: str) -> bool:
    return _check(word)[1]


def foreign(word: str) -> bool:
    return _check(word)[2]


def broken(word: str) -> bool:
    return _check(word)[3]


def page_check(text: str) -> dict:
    """Per page: words; share of them that are syllables, abbreviations left
    out (an all-capital word that is no syllable: HĐQT, CCCD); share of the
    syllables that carry a mark; share of the words that are broken; and how
    many words hold a letter outside the alphabet."""
    words = [w for w in WORD_RE.findall(normal(text)) if len(w) > 1]
    seen = {w: _check(w) for w in set(words)}
    ok = [w for w in words if seen[w][0]]
    counted = sum(1 for w in words if seen[w][0] or not w.isupper())
    return {"words": len(words),
            "syllables": round(len(ok) / counted, 3) if counted else 0.0,
            "marked": round(sum(seen[w][1] for w in ok) / len(ok), 3) if ok else 0.0,
            "broken": round(sum(seen[w][3] for w in words) / len(words), 4) if words else 0.0,
            "foreign": sum(seen[w][2] for w in words)}
