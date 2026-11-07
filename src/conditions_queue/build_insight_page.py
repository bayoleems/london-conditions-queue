"""Build the "London's Conditions Queue" findings page: inject outputs/summary.json into the page template.

Run: python -m conditions_queue.build_insight_page   ->  site/insight.html
"""
import json

from conditions_queue.paths import SITE_INSIGHT_HTML, SUMMARY_JSON, TEMPLATES


def main():
    data = json.loads(SUMMARY_JSON.read_text())
    template = (TEMPLATES / "insight_page.html").read_text()
    SITE_INSIGHT_HTML.parent.mkdir(parents=True, exist_ok=True)
    SITE_INSIGHT_HTML.write_text(template.replace("/*DATA*/null", json.dumps(data)))
    print(f"wrote {SITE_INSIGHT_HTML}")


if __name__ == "__main__":
    main()
