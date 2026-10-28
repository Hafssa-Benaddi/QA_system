"""
Builds urls.txt: the 50+ documentation pages for the project, automatically.
Topic: official Git documentation (https://git-scm.com/docs)

Install:  pip install requests beautifulsoup4
Run:      python collect_urls.py
"""

import re
import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://git-scm.com/docs"
BASE = "https://git-scm.com"
TARGET = 70  # collect a few more than 50, some pages may fail or be too short

# Everyday commands first: they have rich, question-friendly content
PRIORITY = [
    "git-init", "git-clone", "git-add", "git-commit", "git-status", "git-diff",
    "git-log", "git-branch", "git-checkout", "git-switch", "git-restore",
    "git-merge", "git-rebase", "git-stash", "git-reset", "git-revert",
    "git-tag", "git-remote", "git-fetch", "git-pull", "git-push", "git-config",
    "git-cherry-pick", "git-show", "git-blame", "git-bisect", "git-clean",
    "git-rm", "git-mv", "git-reflog", "git-worktree", "git-submodule",
    "git-grep", "git-gc", "git-apply", "git-format-patch", "git-am",
    "git-describe", "git-shortlog", "git-archive", "git-notes", "git-hook",
]


def main():
    html = requests.get(INDEX_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")

    # Every link of the form /docs/git-xxx on the index page
    found = []
    for a in soup.find_all("a", href=True):
        m = re.fullmatch(r"/docs/(git-[a-z0-9-]+)", a["href"])
        if m and m.group(1) not in found:
            found.append(m.group(1))

    print(f"{len(found)} command pages found on the index")

    ordered = [p for p in PRIORITY if p in found]
    ordered += [p for p in found if p not in ordered]
    urls = [f"{BASE}/docs/{name}" for name in ordered[:TARGET]]

    with open("urls.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(urls))
    print(f"{len(urls)} URLs saved in urls.txt")


if __name__ == "__main__":
    main()
