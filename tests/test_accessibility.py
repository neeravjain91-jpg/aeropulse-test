"""Accessibility audit for static/index.html (WCAG 2.1 basics).

Checks buttons have accessible text, navigation tabs have ARIA roles,
no sub-9px fonts, and interactive canvases have aria-labels.
"""
import os
import re
from html.parser import HTMLParser


class A11yParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons = []
        self.canvases = []
        self.images = []
        self.tabs = []
        self.current_button = None
        self.in_button = False

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)

        if tag == "button":
            self.in_button = True
            self.current_button = {"attrs": attr_dict, "text": ""}
        elif tag == "canvas":
            self.canvases.append(attr_dict)
        elif tag == "img":
            self.images.append(attr_dict)

        classes = attr_dict.get("class", "")
        if "side-tab-btn" in classes:
            self.tabs.append(attr_dict)

    def handle_data(self, data):
        if self.in_button and self.current_button is not None:
            self.current_button["text"] += data

    def handle_endtag(self, tag):
        if tag == "button":
            self.in_button = False
            self.buttons.append(self.current_button)
            self.current_button = None


# Canvas IDs that are decorative / part of the 3D engine visualization
# and don't require aria-label (they are visual-only overlays)
_DECORATIVE_CANVAS_IDS = {
    "engineLabelsCanvas",       # leader line overlay (aria-hidden)
    "diagramKinematicsCanvas",  # 2D kinematics diagram
    "diagramValveCanvas",       # valve timing diagram
    "diagramPressureCanvas",    # P-V indicator diagram
}


def test_accessibility():
    html_path = os.path.join(os.path.dirname(__file__), "../static/index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    parser = A11yParser()
    parser.feed(content)

    # 1. All <button> elements have accessible text
    for i, btn in enumerate(parser.buttons):
        text = btn["text"].strip()
        aria_label = btn["attrs"].get("aria-label", "").strip()
        assert text or aria_label, f"Button {i} ({btn['attrs'].get('id', 'no-id')}) missing accessible text"

    # 2. Navigation tabs have proper role attributes
    for i, tab in enumerate(parser.tabs):
        role = tab.get("role", "")
        assert role in ["tab", "navigation", "button", "menuitem"], (
            f"Tab {i} ({tab.get('id', 'no-id')}) missing proper role attribute"
        )

    # 3. No inline font sizes below 9px
    small_fonts = re.findall(r'font-size:\s*[0-8](?:\.[0-9]+)?px', content)
    assert not small_fonts, f"Found font sizes below 9px: {small_fonts}"

    # 4. Interactive canvases have aria-label (skip decorative / aria-hidden)
    for i, cvs in enumerate(parser.canvases):
        if cvs.get("aria-hidden") == "true":
            continue
        if cvs.get("id", "") in _DECORATIVE_CANVAS_IDS:
            continue
        assert "aria-label" in cvs, f"Canvas {i} ({cvs.get('id', 'no-id')}) missing aria-label"

    # 5. All <img> elements have alt text
    for i, img in enumerate(parser.images):
        assert "alt" in img, f"Image {i} ({img.get('id', 'no-id')}) missing alt text"
