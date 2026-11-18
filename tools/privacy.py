"""What keeps a person out of the sets: rules on file names and on page text.

Shared by the set builders (steps 3 to 5), so a rule tuned for one set holds
for the others. Each rule errs toward leaving a page out: abstain over guess.
Text is matched in NFC (syllables.normal): a text layer may keep its marks
apart, and "thường trú" spelt so matches no pattern.
"""
import re

from syllables import normal

# An honorific in a file name: "..._elected_mr__[NAME]_...".
NAMED_RE = re.compile(r"(^|_)(mr|mrs|ms|ong|ba)(_|$)")
# A file name that marks an insider's or related person's trade ("kqgd": trade
# result; "nclq": related person), or ends in a Vietnamese full name: a surname,
# a common middle name, one or two more words ("..._tv_hdqt_[surname]_van_[given].pdf").
# The middle name is what tells a name from a phrase that starts like a surname
# (phan_phoi, dang_ky, giai_trinh); "le" after "dieu_" is the charter (điều lệ).
# Checked on 2,087 file names of the step 0 survey: 42 matches, all names.
SURNAMES = ("nguyen|tran|(?<!dieu_)le|pham|hoang|huynh|phan|vu|vo|dang|bui|do|ho|ngo|duong|ly"
            "|dinh|truong|trinh|mai|cao|ha|lam|luong|ta|doan|vuong|trieu|chu|kieu|la|thai"
            "|quach|luu|tang|to|ton")
MIDDLE_NAMES = ("van|thi|duc|minh|ngoc|quoc|huu|thanh|xuan|hong|kim|cong|quang|trong|tuan|thu"
                "|bao|gia|manh|tien|viet|hai|phuong|trung|the|tan|dinh|dang|anh|hoang|nhu|my"
                "|khanh|hoai|thuy|phu|chi|ba")
PERSON_FILE_RE = re.compile(rf"(^|_)(kqgd|nclq|nguoi_noi_bo|giao_dich_cua)(_|$)"
                            rf"|(^|_)({SURNAMES})_({MIDDLE_NAMES})(_[a-z]+){{1,2}}(_\d)?"
                            rf"(_signed|_sign|_da_ky|_ky)?\.pdf$")

# A label for a person's identity, then a value on the same line; a blank form
# field ("ID No.: ......") has no digits after it. A birth year alone counts too
# ("born in 19xx" in a court ruling on an employee).
IDENTITY_RE = re.compile(r"(\bID\b|identity|identification|passport|citizen|CCCD|CMND"
                         r"|date of birth|birth ?date)[^\n\d]{0,40}\d[\d./-]{5,}"
                         r"|\b(born|birth)\b[^\n\d]{0,20}(19|20)\d\d\b", re.I)
# The same in Vietnamese.
IDENTITY_VI_RE = re.compile(r"(CCCD|CMND|căn cước|chứng minh|hộ chiếu|định danh|ngày sinh"
                            r"|sinh ngày)[^\n\d]{0,40}\d[\d./-]{5,}"
                            r"|\bnăm sinh\b[^\n\d]{0,20}(19|20)\d\d\b", re.I)
# A field label alone: a table of insiders or nominees keeps label and value
# apart, so the patterns above miss its values. "Ngày cấp" (date of issue) is not
# among them: every resolution gives it for the company's registration.
IDENTITY_LABEL_RE = re.compile(r"\b(CMND|CCCD|căn cước|chứng minh nhân dân|hộ chiếu|Giấy NSH"
                               r"|số định danh|thường trú|ID card|passport)\b", re.I)

# A court ruling is about a person (pay, conduct), page after page; the whole
# filing is left out. A meeting pack that reports a lawsuit names one on a few
# of its pages, a ruling on most.
COURT_RE = re.compile(r"\b(nguyên đơn|bị đơn|khởi kiện|plaintiff|defendant|lawsuit)s?\b", re.I)
COURT_SHARE = 0.3


def named_file(name: str) -> bool:
    """The file name names a person or marks a person's trade."""
    name = name.lower()
    return bool(NAMED_RE.search(name) or PERSON_FILE_RE.search(name))


def identity(text: str, labels: bool = True) -> bool:
    """The text holds a filled identity field, or with labels, any identity label."""
    text = normal(text)
    return bool(IDENTITY_RE.search(text) or IDENTITY_VI_RE.search(text)
                or labels and IDENTITY_LABEL_RE.search(text))


def court_ruling(texts: list[str]) -> bool:
    """The pages read as a court ruling; a file with no page read does not."""
    return bool(texts) and sum(bool(COURT_RE.search(normal(t))) for t in texts) >= COURT_SHARE * len(texts)
